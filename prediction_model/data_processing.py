from imports import *

class ElectricityDataset(Dataset):
    def __init__(self, data: np.ndarray, sequence_length: int):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - 96

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + 96, 0]
        if len(y) < 96:
            raise ValueError("Not enough data to create label")
        return torch.FloatTensor(x), torch.FloatTensor(y).unsqueeze(-1)

def preprocess_data(df: pd.DataFrame, logger: logging.Logger) -> tuple[np.ndarray, StandardScaler, np.ndarray]:
    try:
        features = ['import_kwh', 'avg_import_kw', 'power_factor',
                    'phase_a_current', 'phase_a_voltage',
                    'phase_b_current', 'phase_b_voltage',
                    'phase_c_current', 'phase_c_voltage']
        original_import_kwh = df['import_kwh'].copy()
        df['import_kwh_diff'] = df['import_kwh'].diff().fillna(0)
        df['import_kwh'] = df['import_kwh_diff']
        df = df.drop(columns=['import_kwh_diff'])
        df = df[features].ffill().infer_objects(copy=False).fillna(0)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        logger.info("Data preprocessed successfully using differences for import_kwh")
        return scaled_data, scaler, original_import_kwh.values
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
        raise