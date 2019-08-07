import argparse
import glob
import os
import ImgType
import multiprocessing as mp
from typing import List


def get_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument('--data_path', type=str, help='directory containing the skeleton data')
    parser.add_argument('--img_type', type=int, choices=[1, 2, 3], help='Image type to compute'
                                                                   '1 - CaetanoMagnitude (SkeleMotion - AVSS2019)'
                                                                   '2 - CaetanoOrientation (SkeleMotion - AVSS2019)'
                                                                   '3 - CaetanoTSRJI (TSRJI - SIBGRAPI2019)', default=1)
    parser.add_argument('--temp_dist', nargs='+', type=int, help='Temporal distance between frames', default=1)
    parser.add_argument('--path_to_save', type=str, help='directory to save the skeleton images')
    args = parser.parse_args()
    print(args)
    return args


def save_extraction_list(list_extraction: List[str], path_to_save: str = '') -> None:
    file = open(os.path.join(path_to_save, 'extraction_list.txt'), 'w')
    file.writelines("%s\n" % l for l in list_extraction)
    file.close()


def get_skeleton_files(data_path: str) -> List[str]:
    file_list = []
    for file in glob.glob(os.path.join(data_path, '*.skeleton')):
        file_list.append(file)
    return file_list


def check_path(path_to_check: str) -> None:
    try:
        if not os.path.exists(path_to_check):
            print('Creating path: ' + path_to_check)
            os.makedirs(path_to_check)
            print('Path ' + path_to_check + ' OK')
    except OSError:
        print('Error: Creating directory. ' + path_to_check)


def worker(args: tuple):
    obj, method_name, skl_file, path_to_save, temp_dist = args
    getattr(obj, 'set_temporal_scale')(temp_dist)
    ret = getattr(obj, method_name)(skl_file, path_to_save)
    return ret


def main(parser: argparse.ArgumentParser) -> None:
    args = get_arguments(parser)
    check_path(args.path_to_save)
    skl_list = get_skeleton_files(args.data_path)
    obj_list = [ImgType.class_img_types[args.img_type]() for _ in range(0, len(skl_list))]
    pool = mp.Pool(mp.cpu_count())
    list_extraction = pool.map(worker, ((obj, 'process_skl_file', skl_file, args.path_to_save, args.temp_dist) for obj, skl_file in zip(obj_list, skl_list)))
    pool.close()
    pool.join()
    pool.terminate()
    save_extraction_list(list_extraction, args.path_to_save)
    print('End')


if __name__ == '__main__':
    main(argparse.ArgumentParser())