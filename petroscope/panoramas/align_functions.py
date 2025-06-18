
from classes import MatchesData, AlignData

def find_homographies(matches_data: 'MatchesData') -> 'AlignData':
    """
    Find homographies between images based on matching data.
    
    Args:
        matches_data: MatchesData object containing correspondences between images
                      and original sizes.
    
    Returns:
        AlignData: Data object containing image paths, transformations (homographies),
                   reference index for alignment, and inliers.
    """
    import cv2
    import numpy as np
    
    diff_corr = matches_data.matches
    orig_sizes = matches_data.orig_sizes
    img_paths = matches_data.img_paths
    n = len(orig_sizes)
    
    inliers = []
    Hs = [[None] * n for _ in range(n)]
    num_matches = np.zeros((n, n), dtype=int)
    
    # Parameters for filtering matches
    confidence_threshold = 0.95
    min_inliers = 5
    max_inliers = 30
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            corrs = diff_corr.pop(0)
            
            if corrs.shape[0] < min_inliers:
                continue
                
            # Filter by confidence
            corrs = corrs[corrs[:, 4] > confidence_threshold]
            if corrs.shape[0] < min_inliers:
                continue
                
            # Run RANSAC to find homography
            H_ij, mask = cv2.findHomography(
                corrs[:, 0:2], corrs[:, 2:4],
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=1.0,
            )
            if H_ij is None or mask.sum() < min_inliers:
                continue
                
            num_matches[i][j] = int(mask.sum())
            num_matches[j][i] = num_matches[i][j]
            
            # Save forward H, and compute inverse for backward
            Hs[i][j] = H_ij
            try:
                Hs[j][i] = np.linalg.inv(H_ij)
                Hs[j][i] /= Hs[j][i][2, 2]
            except np.linalg.LinAlgError:
                assert "Singular homography matrix"
                
            # Store top-k inliers (sorted by confidence)
            inliers_ij = corrs[mask.ravel().astype(bool)]
            topk = inliers_ij[inliers_ij[:, -1].argsort()[::-1]][:max_inliers]
            topk = topk[:, :-1]  # Remove confidence
            inliers += [[i, j, *inl] for inl in topk]
    
    # Initialize the transformations for each image
    transforms = [np.eye(3) for i in range(n)]
    
    queryIdx = [i for i in range(n)]
    targetIdx = []
    outliersIdx = []
    
    # Choose a pivot image randomly
    pivot = np.random.randint(0, n - 1)
    targetIdx.append(pivot)
    queryIdx.remove(pivot)
    
    while queryIdx:
        a = num_matches[queryIdx, :][:, targetIdx]
        curr = np.argmax(a.sum(axis=1))
        best_neighb = np.argmax(a[curr])
        
        if Hs[queryIdx[curr]][targetIdx[best_neighb]] is None:
            assert num_matches[queryIdx[curr], targetIdx[best_neighb]] == 0, \
                f"None homography, matches = {num_matches[queryIdx[curr], targetIdx[best_neighb]]}"
            transforms[queryIdx[curr]] = None
            outliersIdx.append(queryIdx[curr])
            queryIdx.pop(curr)
            continue
            
        H = (
            transforms[targetIdx[best_neighb]]
            @ Hs[queryIdx[curr]][targetIdx[best_neighb]]
        )
        H /= H[2, 2]
        transforms[queryIdx[curr]] = H
        targetIdx.append(queryIdx[curr])
        queryIdx.pop(curr)
    
    assert (set(targetIdx) | set(outliersIdx)) == set(range(n)), "Something wrong in lists"
    outliers_mask = [1 if i in outliersIdx else 0 for i in range(n)]
    idx_shift = np.cumsum(outliers_mask)
    
    real_transforms = [transforms[i] for i in targetIdx]
    real_inliers = []
    for inl in inliers:
        if inl[0] in outliersIdx or inl[1] in outliersIdx:
            continue
        inl[0], inl[1] = inl[0] - idx_shift[inl[0]], inl[1] - idx_shift[inl[1]]
        real_inliers.append(inl)
    pivot -= idx_shift[pivot]
    
    # Use the full list of img_paths from input
    return AlignData(img_paths, real_transforms, pivot, real_inliers)
