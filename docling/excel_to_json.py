import pandas as pd
import json
import sys
import os
from pathlib import Path


def xlsx_to_json(
    xlsx_file_path, json_file_path=None, sheet_name=None, orient="records"
):
    """
    Convert an Excel (.xlsx) file to JSON format.

    Parameters:
    xlsx_file_path (str): Path to the input Excel file
    json_file_path (str, optional): Path for the output JSON file. If None, uses same name as input with .json extension
    sheet_name (str/int, optional): Name or index of sheet to convert. If None, converts the first sheet
    orient (str): JSON orientation ('records', 'index', 'values', 'split', 'table')

    Returns:
    str: Path to the created JSON file
    """

    try:
        # Check if input file exists
        if not os.path.exists(xlsx_file_path):
            raise FileNotFoundError(f"Input file '{xlsx_file_path}' not found.")

        # Generate output filename if not provided
        if json_file_path is None:
            input_path = Path(xlsx_file_path)
            json_file_path = input_path.with_suffix(".json")

        # Read Excel file
        print(f"Reading Excel file: {xlsx_file_path}")

        if sheet_name is not None:
            df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)
            print(f"Converting sheet: {sheet_name}")
        else:
            df = pd.read_excel(xlsx_file_path)
            print("Converting first sheet")

        # Convert DataFrame to JSON
        print(f"Converting to JSON format (orient='{orient}')...")
        json_data = df.to_json(orient=orient, date_format="iso", indent=2)

        # Parse and re-format for pretty printing
        parsed_json = json.loads(json_data)

        # Write to JSON file
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(parsed_json, json_file, indent=2, ensure_ascii=False)

        print(f"Successfully converted '{xlsx_file_path}' to '{json_file_path}'")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")

        return str(json_file_path)

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return None


def list_excel_sheets(xlsx_file_path):
    """
    List all sheet names in an Excel file.

    Parameters:
    xlsx_file_path (str): Path to the Excel file

    Returns:
    list: List of sheet names
    """
    try:
        excel_file = pd.ExcelFile(xlsx_file_path)
        return excel_file.sheet_names
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return []


def main():
    """
    Main function with hardcoded file paths.
    """
    # =================== MODIFY THESE PATHS ===================
    # Change these paths to point to your actual files

    # For Windows (use one of these formats):
    # xlsx_file = r"C:\Users\YourName\Documents\your_file.xlsx"
    # xlsx_file = "C:\\Users\\YourName\\Documents\\your_file.xlsx"
    # xlsx_file = "C:/Users/YourName/Documents/your_file.xlsx"

    # For Mac/Linux:
    xlsx_file = "/mnt/c/AI/CRMRecords/example.xlsx"
    # xlsx_file = "~/Documents/your_file.xlsx"

    # Output JSON file (optional - set to None for auto-generate)
    json_file = (
        None  # This will create a .json file with the same name as the Excel file
    )
    # json_file = r"C:\Users\YourName\Documents\output.json"  # Specify custom output path

    # Sheet to convert (optional - set to None for first sheet)
    sheet_name = None  # Use first sheet
    # sheet_name = "Sheet1"  # Use specific sheet name
    # sheet_name = 0  # Use sheet by index (0 = first sheet)

    # JSON format orientation
    orient = "records"  # Most common format: [{'col1': val1, 'col2': val2}, ...]
    # orient = 'index'    # {'index1': {'col1': val1, 'col2': val2}, ...}
    # orient = 'values'   # [[val1, val2, ...], ...]

    # =========================================================

    # Check if file exists before proceeding
    if not os.path.exists(xlsx_file):
        print(f"❌ Error: File not found at '{xlsx_file}'")
        print("Please update the xlsx_file path in the main() function.")
        return

    # List available sheets
    print("📊 Available sheets in the Excel file:")
    sheets = list_excel_sheets(xlsx_file)
    if sheets:
        for i, sheet in enumerate(sheets):
            print(f"  {i}: {sheet}")
        print()

    # Convert file
    print("🔄 Starting conversion...")
    result = xlsx_to_json(xlsx_file, json_file, sheet_name, orient)

    if result:
        print(f"\n✅ Conversion completed successfully!")
        print(f"📁 Output file: {result}")
    else:
        print("\n❌ Conversion failed!")


# Alternative function for converting all sheets
def xlsx_to_json_all_sheets(xlsx_file_path, output_dir=None):
    """
    Convert all sheets in an Excel file to separate JSON files.

    Parameters:
    xlsx_file_path (str): Path to the input Excel file
    output_dir (str, optional): Directory for output files. If None, uses same directory as input

    Returns:
    list: List of created JSON file paths
    """
    try:
        if output_dir is None:
            output_dir = Path(xlsx_file_path).parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        excel_file = pd.ExcelFile(xlsx_file_path)
        json_files = []

        for sheet_name in excel_file.sheet_names:
            # Create output filename
            safe_sheet_name = "".join(
                c for c in sheet_name if c.isalnum() or c in (" ", "-", "_")
            ).rstrip()
            json_filename = f"{Path(xlsx_file_path).stem}_{safe_sheet_name}.json"
            json_file_path = output_dir / json_filename

            # Convert sheet
            df = pd.read_excel(xlsx_file_path, sheet_name=sheet_name)
            json_data = df.to_json(orient="records", date_format="iso", indent=2)
            parsed_json = json.loads(json_data)

            with open(json_file_path, "w", encoding="utf-8") as json_file:
                json.dump(parsed_json, json_file, indent=2, ensure_ascii=False)

            json_files.append(str(json_file_path))
            print(f"Converted sheet '{sheet_name}' to '{json_file_path}'")

        return json_files

    except Exception as e:
        print(f"Error converting all sheets: {str(e)}")
        return []


def convert_all_sheets_example():
    """
    Example function to convert all sheets to separate JSON files.
    """
    # Modify this path to your Excel file
    xlsx_file = r"C:\Users\YourName\Documents\your_file.xlsx"

    # Optional: specify output directory
    output_directory = r"C:\Users\YourName\Documents\json_output"

    print("Converting all sheets to separate JSON files...")
    json_files = xlsx_to_json_all_sheets(xlsx_file, output_directory)

    if json_files:
        print(f"\n✅ Successfully converted {len(json_files)} sheets:")
        for file in json_files:
            print(f"  📁 {file}")
    else:
        print("\n❌ Failed to convert sheets!")


if __name__ == "__main__":
    # Choose which function to run:

    main()  # Convert single sheet

    # OR uncomment the line below to convert all sheets instead:
    # convert_all_sheets_example()
