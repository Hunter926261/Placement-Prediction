import pandas as pd
from src.config import RAW_DATA_PATH


def load_data(RAW_DATA_PATH):
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print("✅ Data loaded successfully")
        return df

    except FileNotFoundError:
        raise FileNotFoundError(
            f"❌ File not found at path: {RAW_DATA_PATH}"
        )

    except Exception as e:
        raise Exception(
            f"❌ Error while loading data: {str(e)}"
        )
