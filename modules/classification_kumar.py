from .features_letters import (
    prune_spurs,
    count_endpoints,
    count_vertical_lines,
    count_horizontal_lines,
    hole_count_and_largest_pct,
    horizontal_symmetry_tb_balance,
    vertical_symmetry_score,
    concavity_tb_strength,
    concavity_lr_strength,
    count_holes
)

def blob_side_from_pct(hole_count: int, hole_pct: float, split: float = 23.0) -> str:
    if hole_count >= 2:
        return "B"
    return "DOQ" if hole_pct >= split else "ABPR"


def classify_blob_branch_cramm(A01, skel01, blob_split=30.0, sym_thresh=0.60):
    hole_count, hole_pct = hole_count_and_largest_pct(A01)
    if hole_count == 0:
        return "NO_BLOB"

    sk = prune_spurs(skel01, max_length=2)
    e = count_endpoints(sk)

    side = blob_side_from_pct(hole_count, hole_pct, split=blob_split)

    if side == "B":
        return "B"

    if side == "ABPR":
        if hole_count >= 2:
            return "B"
        if e == 1:
            return "P"
        if e == 2:
            sym = vertical_symmetry_score(A01)
            return "A" if sym >= sym_thresh else "R"
        return "UNKNOWN_BLOB"

    # DOQ side
    if e >= 1:
        return "Q"

    sym = vertical_symmetry_score(A01)
    return "O" if sym >= sym_thresh else "D"


def classify_no_blob_cramm(A01, skel01, prune_len=2, sym_thresh=0.70):
    sk = prune_spurs(skel01, max_length=prune_len)
    e = count_endpoints(sk)

    # endpoints = 3 -> E, F, T, Y
    if e == 3:
        hs = horizontal_symmetry_tb_balance(A01)
        if hs >= sym_thresh:
            return "E"

        hstroke = count_horizontal_lines(sk)
        if hstroke <= 0:
            return "Y"
        if hstroke == 1:
            return "T"
        return "F"

    # endpoints = 2
    if e == 2:
        vstroke = count_vertical_lines(sk, min_frac=0.35)
        hstroke = count_horizontal_lines(sk)
        sym_tb = horizontal_symmetry_tb_balance(A01)
        vs = vertical_symmetry_score(A01)
        north, south = concavity_tb_strength(A01)
        west, east = concavity_lr_strength(A01)

        if vs > 0.75:
            if sym_tb >= sym_thresh:
                return "I"

            if north > south + 0.03:
                return "M"

            if vstroke >= 4:
                return "W"

            if south > north + 0.03:
                return "U"

            return "V"

        else:
            if vstroke == 0:
                return "C" if sym_tb >= sym_thresh else "S"

            if vstroke == 1:
                if hstroke == 1:
                    return "L" if east > west + 0.03 else "G"
                if hstroke == 2:
                    return "Z"
                if hstroke == 0:
                    return "J"

            if vstroke >= 2:
                return "N"

    # endpoints = 4 -> K, H, X
    if e == 4:
        vstroke = count_vertical_lines(sk, min_frac=0.35)
        hstroke = count_horizontal_lines(sk)

        if vstroke == 1:
            return "K"
        else:
            if hstroke == 1:
                return "H"
            if hstroke == 0:
                return "X"

def classify_letter(A01, skel01):
    if count_holes(A01) > 0:
        return classify_blob_branch_cramm(A01, skel01)
    return classify_no_blob_cramm(A01, skel01)

