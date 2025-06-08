import requests

main_pdf_folder = 'pdfs\\'


def get_pdf_name(puzzle_id: int):
    return f"{main_pdf_folder}sudoku_{puzzle_id}.pdf"


def get_puzzle(puzzle_id: int):
    url = f"https://www.dailykillersudoku.com/pdfs/{puzzle_id}.pdf"
    response = requests.get(url)
    filename = get_pdf_name(puzzle_id)
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename
