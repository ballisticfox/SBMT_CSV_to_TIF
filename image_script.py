import argparse
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree

# Optional
try:
    import torch
    _has_torch = True
except ImportError:
    _has_torch = False
try:
    from tqdm import tqdm
    _has_tqdm = True
except ImportError:
    _has_tqdm = False

def normalize_longitude(lon):
    return ((lon + 180.0) % 360.0) - 180.0

def latlon_to_pixel(lat, lon, W, H):
    x = (normalize_longitude(lon) + 180) / 360 * (W - 1)
    y = (90 - lat) / 180 * (H - 1)
    return x, y

def scale_to_uint(arr, bits=8, vmin=None, vmax=None):
    a = arr.astype(float).copy()
    if vmin is None: vmin = np.nanmin(a)
    if vmax is None: vmax = np.nanmax(a)
    if bits == 8:
        maxv = 255
        dtype = np.uint8
    else:
        maxv = 65535
        dtype = np.uint16
    a = (a - vmin) / (vmax - vmin)
    a = np.clip(a, 0, 1)
    out = (a * maxv).astype(dtype)
    out[np.isnan(arr)] = 0
    return out


def _interp_chunk(args):
    method, pts_u, vals, tree, mask_slice, k, radius = args
    if method == "nearest":
        _, idx = tree.query(pts_u, k=1)
        return vals[idx]
    elif method in ("bilinear", "idw"):
        d, idx = tree.query(pts_u, k=k, distance_upper_bound=radius or np.inf)
        d[d == np.inf] = 1e-6
        w = 1 / d
        w /= w.sum(axis=1, keepdims=True)
        return np.sum(vals[idx] * w, axis=1)
    return None

def interpolate_local(img, method="nearest", k=4, radius=None, gpu=None, batch_size=20000):
    """
    Local neighbor interpolation for large grids.
    - img: 2D np.ndarray with np.nan for missing values
    - method: "nearest" or "idw"
    - k: number of nearest neighbors
    - gpu: None, "metal", or "cuda"
    """
    mask = np.isnan(img)
    if not mask.any():
        return img

    H, W = img.shape
    pad = W // 8
    img_pad = np.hstack([img[:, -pad:], img, img[:, :pad]])
    mask_pad = np.isnan(img_pad)

    yy, xx = np.indices(img_pad.shape)
    known = ~mask_pad
    pts_known = np.column_stack((yy[known], xx[known]))
    vals_known = img_pad[known]

    pts_unknown = np.column_stack((yy[mask_pad], xx[mask_pad]))

    # GPU device
    use_gpu = gpu in ("metal", "cuda") and _has_torch
    if use_gpu:
        device = torch.device("mps" if gpu=="metal" else "cuda")
        vals_known_t = torch.tensor(vals_known, dtype=torch.float32, device=device)
        results = torch.empty(len(pts_unknown), dtype=torch.float32, device=device)
    else:
        results = np.empty(len(pts_unknown), dtype=np.float32)

    tree = cKDTree(pts_known)

    n_batches = (len(pts_unknown) + batch_size - 1) // batch_size
    pbar = tqdm(total=n_batches, desc=f"Interpolating ({method})")

    for i in range(n_batches):
        start = i*batch_size
        end = min((i+1)*batch_size, len(pts_unknown))
        batch_pts = pts_unknown[start:end]

        # query k nearest neighbors
        dists, idxs = tree.query(batch_pts, k=k, distance_upper_bound=radius or np.inf)

        # handle missing neighbors
        dists = np.where(np.isinf(dists), 1e-6, dists)

        if method == "nearest":
            nearest_idx = idxs[:, 0] if k > 1 else idxs
            if use_gpu:
                results[start:end] = vals_known_t[torch.tensor(nearest_idx, device=device)]
            else:
                results[start:end] = vals_known[nearest_idx]

        elif method in ("bilinear", "idw"):
            # inverse distance weighting
            w = 1.0 / dists
            w /= w.sum(axis=1, keepdims=True)
            if use_gpu:
                vals_expand = vals_known_t[torch.tensor(idxs, device=device)]
                w_t = torch.tensor(w, dtype=torch.float32, device=device)  # <-- force float32 for Metal
                results[start:end] = (vals_expand * w_t).sum(dim=1)
            else:
                results[start:end] = np.sum(vals_known[idxs] * w, axis=1)

        pbar.update(1)

    pbar.close()

    # fill interpolated values
    img_pad[mask_pad] = results if not use_gpu else results.cpu().numpy()
    return img_pad[:, pad:-pad]

def interpolate(img, method="nearest", k=4, radius=None, workers=None):
    mask = np.isnan(img)
    if not np.any(mask):
        return img

    H, W = img.shape

    # Wrap horizontally for longitude continuity
    pad = W // 8
    img_pad = np.hstack([img[:, -pad:], img, img[:, :pad]])
    mask_pad = np.isnan(img_pad)

    yy, xx = np.indices(img_pad.shape)
    known = ~mask_pad
    pts = np.column_stack((yy[known], xx[known]))
    vals = img_pad[known]

    tree = cKDTree(pts)
    pts_u = np.column_stack((yy[mask_pad], xx[mask_pad]))

    if method in ("nearest", "bilinear", "idw"):
        chunk = 20000
        tasks = []
        for i in range(0, len(pts_u), chunk):
            j = i + chunk
            tasks.append((method, pts_u[i:j], vals, tree, mask_pad, k, radius))

        pbar = tqdm(total=len(tasks), desc=f"Interpolating ({method})")

        results = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for res in ex.map(_interp_chunk, tasks):
                results.append(res)
                pbar.update(1)

        all_vals = np.concatenate(results)
        img_pad[mask_pad] = all_vals
    return img_pad[:, pad:-pad]

  
def build_layer(df, col, W, H, weight_col="Facet Area", interp="nearest", gpu=None):
    lat, lon = df["Latitude"].values, df["Longitude"].values
    vals = df[col].astype(float).values
    weights = df[weight_col].astype(float).values if weight_col in df else np.ones_like(vals)

    xs, ys = latlon_to_pixel(lat, lon, W, H)
    xi, yi = np.round(xs).astype(int), np.round(ys).astype(int)
    xi = np.clip(xi, 0, W - 1)
    yi = np.clip(yi, 0, H - 1)

    arr, wgt = np.zeros((H, W)), np.zeros((H, W))

    iterator = zip(xi, yi, vals, weights)
    if _has_tqdm:
        iterator = tqdm(iterator, total=len(vals), desc=f"Accumulating {col}")

    for x, y, v, w in iterator:
        arr[y, x] += v * w
        wgt[y, x] += w

    arr[wgt > 0] /= wgt[wgt > 0]
    arr[wgt == 0] = np.nan

    if gpu in ("metal", "cuda"):
        return interpolate_local(arr, method=interp, k=4, gpu=gpu)
    else:
        return interpolate(arr, interp, workers=8)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--col", default="Albedo")
    ap.add_argument("--rgb", help="3 comma-separated columns for RGB")
    ap.add_argument("--weight", default="Facet Area")
    ap.add_argument("--interp", choices=["none","nearest","bilinear"], default="nearest")
    ap.add_argument("--bits", type=int, choices=[8,16], default=8)
    ap.add_argument("--gpu", choices=["metal","cuda"], help="Use GPU acceleration (Apple Metal or CUDA)")
    ap.add_argument("--out", type=Path, default=Path("out.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv, comment="#", skip_blank_lines=True, skiprows=[1])

    if args.rgb:
        cols = [c.strip() for c in args.rgb.split(",")]
        layers = [build_layer(df, c, args.width, args.height, args.weight, args.interp, args.gpu) for c in cols]
        bands = [scale_to_uint(L, args.bits) for L in layers]
        rgb = np.stack(bands, axis=-1)
        Image.fromarray(rgb).save(args.out)
    else:
        L = build_layer(df, args.col, args.width, args.height, args.weight, args.interp, args.gpu)
        img = scale_to_uint(L, args.bits)
        Image.fromarray(img).save(args.out)

if __name__ == "__main__":
    main()
# python image_script.py phobos_low.csv --col Albedo --bits 16 --width 2048 --height 1024 --interp bilinear --gpu metal --out albedo_low.tif 
# python image_script.py phobos_high.csv --col Albedo --bits 16 --width 16384 --height 8192 --interp nearest --gpu metal --out albedo_high.tif 
# python image_script.py phobos_high.csv --col Albedo --bits 16 --width 16384 --height 8192 --interp bilinear --gpu metal --out albedo_high_bi.tif 
# python image_script.py phobos_high.csv --col Radius --bits 16 --width 16384 --height 8192 --interp bilinear --gpu metal --out height_high_bi.tif 
