#!/usr/bin/env python3

import math, sys, argparse

def parse_command_line():
    parser = argparse.ArgumentParser(
        description="""
┌─────────────────────────────────────────────────────────────┐
│         STAR File Nucleosome Orientation Comparator         │
├─────────────────────────────────────────────────────────────┤
│ This script compares two nucleosome orientation STAR files  │
│ and separates particles by angular distance threshold.      │
└─────────────────────────────────────────────────────────────┘
""",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'faked_star',
        type=str,
        help='Input STAR file: Faked nucleosome orientation'
    )
    parser.add_argument(
        'recalculated_star',
        type=str,
        help='Input STAR file: Recalculated nucleosome orientation'
    )
    parser.add_argument(
        'angle_threshold',
        type=float,
        help='Threshold for maximum allowed angular distance (degrees)'
    )
    parser.add_argument(
        'Cn_symmetry',
        type=int,
        help='N for Cn symmetry (e.g., 2 for C2, 4 for C4)'
    )
    parser.add_argument(
        'output_prefix',
        type=str,
        help='Prefix for output STAR files'
    )

    parser.epilog = """
Example usage:
  python compare_star.py faked.star recalculated.star 5.0 2 output

Output:
  output_lessthan5.0_change_line.star
  output_morethan5.0_change_line.star
  output (log of angular distances)

For help:
  python compare_star.py --help
"""

    args = parser.parse_args()
    return args.faked_star, args.recalculated_star, args.angle_threshold, args.Cn_symmetry, args.output_prefix

def main():
    faked_star, recalculated_star, threshold, sym_n, output_prefix = parse_command_line()

    f1 = open(faked_star, "r")
    faked_lines = f1.readlines()
    f2 = open(recalculated_star, "r")
    recalculated_lines = f2.readlines()

    output_less = output_prefix + "_lessthan" + str(threshold) + "_change_line.star"
    output_more = output_prefix + "_morethan" + str(threshold) + "_change_line.star"
    out_less = open(output_less, "w")
    out_more = open(output_more, "w")
    out_log = open(output_prefix, "w")

    is_fake_v30 = judge_relion30_or_relion31(faked_lines)
    mline1 = judge_mline0(faked_lines) if is_fake_v30 else judge_mline1(faked_lines, judge_mline0(faked_lines))
    if mline1 < 0:
        print("Faked STAR format error.")
        quit()

    is_recal_v30 = judge_relion30_or_relion31(recalculated_lines)
    mline2 = judge_mline0(recalculated_lines) if is_recal_v30 else judge_mline1(recalculated_lines, judge_mline0(recalculated_lines))
    if mline2 < 0:
        print("Recalculated STAR format error.")
        quit()

    for i in range(mline1):
        out_less.write(faked_lines[i])
        out_more.write(faked_lines[i])
        if faked_lines[i].split():
            if faked_lines[i].split()[0] == "_rlnAngleRot":
                Rot_index = int(faked_lines[i].split('#')[1]) - 1
            if faked_lines[i].split()[0] == "_rlnAngleTilt":
                Tilt_index = int(faked_lines[i].split('#')[1]) - 1
            if faked_lines[i].split()[0] == "_rlnImageName":
                IMG_index = int(faked_lines[i].split('#')[1]) - 1

    for i in range(mline2):
        if recalculated_lines[i].split():
            if recalculated_lines[i].split()[0] == "_rlnAngleRot":
                Rot2_index = int(recalculated_lines[i].split('#')[1]) - 1
            if recalculated_lines[i].split()[0] == "_rlnAngleTilt":
                Tilt2_index = int(recalculated_lines[i].split('#')[1]) - 1
            if recalculated_lines[i].split()[0] == "_rlnImageName":
                IMG2_index = int(recalculated_lines[i].split('#')[1]) - 1

    faked_records = []
    for i in range(mline1, len(faked_lines)):
        if faked_lines[i].split():
            name = faked_lines[i].split()[IMG_index]
            serial = int(name.split('@')[0])
            filename = "_".join(extract_filename(name))
            faked_records.append((serial, filename, i))

    recalculated_records = []
    for i in range(mline2, len(recalculated_lines)):
        if recalculated_lines[i].split():
            name = recalculated_lines[i].split()[IMG2_index]
            serial = int(name.split('@')[0])
            filename = "_".join(extract_filename(name))
            recalculated_records.append((serial, filename, i))

    for serial1, name1, idx1 in faked_records:
        for serial2, name2, idx2 in recalculated_records:
            if serial1 == serial2 and name1 == name2:
                rot1 = float(faked_lines[idx1].split()[Rot_index])
                tilt1 = float(faked_lines[idx1].split()[Tilt_index])
                rot2 = float(recalculated_lines[idx2].split()[Rot2_index])
                tilt2 = float(recalculated_lines[idx2].split()[Tilt2_index])

                min_value = min(
                    calculateAngularDistance(*(VECTOR3D(rot1 + n * 360.0 / sym_n, tilt1, 0.0) + VECTOR3D(rot2, tilt2, 0.0)))
                    for n in range(sym_n)
                )

                out_log.write(f"{serial1}@{name1}\t{min_value:.3f}\n")
                if min_value <= threshold:
                    out_less.write(faked_lines[idx1])
                else:
                    out_more.write(faked_lines[idx1])

                break

    f1.close()
    f2.close()
    out_less.close()
    out_more.close()
    out_log.close()
    print("Done. Results written to:")
    print(f"  {output_less}")
    print(f"  {output_more}")
    print(f"  {output_prefix} (log)")

def extract_filename(imagename):
    return imagename.split('@')[1].split('_')[1:]

def VECTOR3D(x, y, z):
    return [x, y, z]

def Euler_angles2matrix(alpha, beta, gamma):
    alpha, beta, gamma = map(DEG2RAD, [alpha, beta, gamma])
    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sa, sb, sg = math.sin(alpha), math.sin(beta), math.sin(gamma)
    return [
        [cg * cb * ca - sg * sa, cg * cb * sa + sg * ca, -cg * sb],
        [-sg * cb * ca - cg * sa, -sg * cb * sa + cg * ca, sg * sb],
        [sb * ca, sb * sa, cb]
    ]

def DEG2RAD(x):
    return x * math.pi / 180.0

def calculateAngularDistance(rot1, tilt1, psi1, rot2, tilt2, psi2):
    E1 = Euler_angles2matrix(rot1, tilt1, psi1)
    E2 = Euler_angles2matrix(rot2, tilt2, psi2)
    axes_dist = sum(ACOSD(CLIP(dotProduct(E1[i], E2[i]), -1, 1)) for i in range(3)) / 3.0
    return axes_dist

def dotProduct(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def ACOSD(x):
    return math.acos(x) * 180.0 / math.pi

def CLIP(x, a, b):
    return max(min(x, b), a)

def judge_relion30_or_relion31(inline):
    for line in inline[:3]:
        if line.strip().startswith("#") and len(line.split()) >= 3 and int(line.split()[2]) > 30000:
            return False
    return True

def judge_mline0(inline):
    intarget = False
    for i, line in enumerate(inline[:60]):
        if not line.strip():
            continue
        if line.strip().startswith("_"):
            intarget = True
        elif intarget:
            return i
    return -1

def judge_mline1(inline, start):
    intarget = False
    for i, line in enumerate(inline[start:start + 70], start=start):
        if not line.strip():
            continue
        if line.strip().startswith("_"):
            intarget = True
        elif intarget:
            return i
    return -1

if __name__ == "__main__":
    main()
