from sympy import plot

from .features_letters import (
    count_holes, prune_spurs, count_endpoints,
    count_vertical_lines, count_horizontal_lines,
    hole_count_and_largest_pct,
    vertical_symmetry_lr_balance,
    horizontal_symmetry_tb_balance,
    vertical_symmetry_score,
    concavity_tb_strength,
    concavity_lr_strength,
    count_vertical_strokes, count_horizontal_strokes, bottom_width_ratio, side_open_score, count_branchpoints,
    center_density_ratio, endpoints_xy
)

def blob_side_from_pct(hole_count: int, hole_pct: float, split: float = 40.0 ) -> str: #float was 33
    print(f"hole_pct={hole_pct}")
    if hole_count >= 2:
        return "B"
    return "DOQ" if hole_pct >= split else "ABPR"

def classify_blob_branch_cramm(A01, skel01, blob_split=35.0, sym_thresh=0.80): #float was 33
    hole_count, hole_pct = hole_count_and_largest_pct(A01)
    if hole_count == 0:
        return "NO_BLOB"

    sk = prune_spurs(skel01, max_length=2)
    e = count_endpoints(sk)

    side = blob_side_from_pct(hole_count, hole_pct, split=blob_split)

    if side == "B":
        return "B"

    if side == "ABPR":
        if e >= 4:
            north, south = concavity_tb_strength(A01)
            vs = vertical_symmetry_lr_balance(A01)  # optional but helpful

            print(f"[e>=5] e={e} north={north:.3f} south={south:.3f} vs={vs:.3f}")

            # M: top concavity stronger (the "dip" at the top middle)
            if south > north and (south - north) > 0.30:
                return "W"
            else:
                return "M"
        if e == 1:
            return "P"
        sym = vertical_symmetry_score(A01)
        return "R" if sym <= 0.63 and e == 2 else "A"

    # DOQ
    vstrokes = count_vertical_strokes(A01, min_frac=0.6)
    if e == 2:
            if vstrokes >=2 :
                if bottom_width_ratio(A01, band_frac=0.20) >= 0.85:
                    return "D"
                else:
                    return "Q"
    if e == 1:
            return "P"
    else:
        return "O" 

import numpy as np



def classify_no_blob_cramm(A01, skel01,
                           prune_len=2,
                           sym_thresh=0.88):
    sk = prune_spurs(skel01, max_length=prune_len)
    e = count_endpoints(sk)

    
    if e >= 5:
        north, south = concavity_tb_strength(A01)
        vs = vertical_symmetry_lr_balance(A01)  # optional but helpful

        print(f"[e>=5] e={e} north={north:.3f} south={south:.3f} vs={vs:.3f}")

        # M: top concavity stronger (the "dip" at the top middle)
        if south > north and (south - north) > 0.10:
                return "W"
        else:
            return "M"


    # endpoints=4 -> H,K,X
    if e >= 4:
        vstroke = count_vertical_strokes(A01, min_frac=0.7)  # <-- key change
        vs = vertical_symmetry_lr_balance(A01)
        hs = horizontal_symmetry_tb_balance(A01)
        if vstroke == 1:
            if vs > 0.6:
                return "I"
            else:
                return "K"
        else:   
            h = count_horizontal_lines(sk)
            bp = count_branchpoints(sk)
            if h == 0:
                r = center_density_ratio(A01, frac=0.35)
                return "X" if r > 0.2 else "N"   # threshold to tune
            if vstroke >= 2:
                if vs >= 0.65:
                    return "H"
                else:
                    return "N"


    # endpoints=3 -> E,F,T,Y
    if e == 3:
        hs = horizontal_symmetry_tb_balance(A01)
        vs = vertical_symmetry_lr_balance(A01)
        vstrokes = count_vertical_strokes(A01, min_frac=0.6)
        if hs >= sym_thresh:
            if vs > 0.50:
                return "V"
            else:
                return "E"

        h = count_horizontal_lines(sk)
        print("HORIZONTALCOUNT:", h)
        if h == 1:
            return "T"
        if h == 0:
            return "Y"
        # 2 or more
        return "F"

    # endpoints=2 -> big set (we’ll refine step-by-step)
    if e == 2:
        vs = round(vertical_symmetry_lr_balance(A01), 2)
        hs = round(horizontal_symmetry_tb_balance(A01), 2)
        north, south = concavity_tb_strength(A01)
        vstroke = count_vertical_strokes(A01, min_frac=0.7)
        hstroke = count_horizontal_strokes(A01, min_frac=0.8)
        west, east = concavity_lr_strength(A01)
        left_ratio, right_ratio = side_open_score(A01)

        print(f"[e==2] VS={vs:.3f} HS={hs:.3f} vstroke={vstroke} north={north:.3f} south={south:.3f}")

        # ---------------------------
        # STEP 1: vertical symmetry split
        # ---------------------------
        if vs >= 0.70:
            
            bw = bottom_width_ratio(A01, band_frac=0.20)
            print(f"[e==2|SYM] bottom_width_ratio={bw:.3f}")
            return "U"


            # ---------------------------
            # STEP 4: W by many vertical strokes
            # ---------------------------
        # ASYM group (later)
        else:
            if vstroke == 0:
                    pts = endpoints_xy(skel01)
                    if len(pts) != 2:
                        return None

                    (x1, y1), (x2, y2) = pts
                    dx = x2 - x1
                    dy = y2 - y1
                    if dx == 0 or dy == 0:
                        return None  # vertical/horizontal weird case

                    # sign of slope (dx*dy)
                    return "Z" if dx * dy > 0 else "S"
            if vstroke == 1:
                if hstroke >= 1:

                    # heuristic starting point
                    # - G: bigger east variation (notch/opening + inner bar)
                    # - L: left boundary straighter (low west), east variation often smaller
                    if hs >= 0.95:
                        return "C"
                    else:
                        bw = bottom_width_ratio(A01, band_frac=0.20)
                        
                        if vs < 0.50:
                            if left_ratio < right_ratio - 0.05:
                                return "J"   # left side more empty
                            elif right_ratio < left_ratio - 0.05:
                                return "L"   
                        # Only call it G if it really looks like G
                        else:
                            return "G"

                               
def classify_letter(A01, skel01):
    if count_holes(A01) > 0:
        return classify_blob_branch_cramm(A01, skel01)
    return classify_no_blob_cramm(A01, skel01)

