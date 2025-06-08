from pdf2image import convert_from_path
import os

main_image_folder = 'images\\'
converted_pdfs_subfolder = 'converted_pdfs\\'


def convert(path: str, sudoku_number: int):
    images = convert_from_path(path, dpi=300)
    filename = "sudoku_" + str(sudoku_number) + ".png"
    fullpath = f"{main_image_folder}{converted_pdfs_subfolder}{filename}"
    if not os.path.exists(fullpath):
        images[0].save(fullpath, "PNG")
    return fullpath, filename
