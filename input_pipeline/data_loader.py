from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pathlib
from scipy.stats import zscore
import torch
import gin
import pytorch_lightning as pl


@gin.configurable
class HAPTDataset(Dataset):
    def __init__(self, run_paths, DATADIR, win_shift, win_size, split="test"):
        self.haptpath = pathlib.Path(run_paths["path_HAPT"])
        self.root_dir = pathlib.Path(DATADIR)
        self.split = split
        self.labels_path = self.root_dir / "labels.txt"
        csv_path_str = self.split + "_csv.csv"
        self.csv_path = self.haptpath / csv_path_str
        self.win_shift = win_shift
        self.win_size = win_size

        if self.csv_path.is_file():
            self.X, self.y = self._create_window(self.csv_path)
        else:
            self._load_dataset()
            self.X, self.y = self._create_window(self.csv_path)

        self._n_timesteps = self.X.shape[1]
        self._n_features = self.X.shape[-1]

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    @property
    def n_timesteps(self):
        return self._n_timesteps

    @property
    def n_features(self):
        return self._n_features

    @property
    def n_labels(self):
        return torch.unique(self.y).size(dim=0)

    def _load_dataset(self):
        raw_label_data = np.loadtxt(fname=self.labels_path, dtype=int)
        label_sections = []
        for value, index, count in zip(
            *np.unique(
                raw_label_data[:, :2], axis=0, return_counts=True, return_index=True
            )
        ):
            label_sections.append(
                {
                    "experiment": value[0],
                    "user": value[1],
                    "data": raw_label_data[index : index + count, 2:],
                }
            )

        accelerometer_files, gyroscope_files = [], []
        for sensor_data_file_path in [
            path for path in self.root_dir.iterdir() if path.name != "labels.txt"
        ]:
            sensor, experiment, user = sensor_data_file_path.stem.split(sep="_")
            if sensor == "acc":
                accelerometer_files.append(
                    {
                        "sensor": sensor,
                        "experiment": int(experiment[3:]),
                        "user": int(user[4:]),
                        "file_path": sensor_data_file_path,
                    }
                )
            elif sensor == "gyro":
                gyroscope_files.append(
                    {
                        "sensor": sensor,
                        "experiment": int(experiment[3:]),
                        "user": int(user[4:]),
                        "file_path": sensor_data_file_path,
                    }
                )

        data_train, data_val, data_test = [], [], []
        training_user_numbers = range(1, 22)
        val_user_numbers = range(28, 31)
        test_user_numbers = range(22, 28)
        for accelerometer_file in accelerometer_files:
            # Find file containing the data of the gyroscope sensor of the same experiment
            gyroscope_file = next(
                gyroscope_file
                for gyroscope_file in gyroscope_files
                if gyroscope_file["experiment"] == accelerometer_file["experiment"]
            )
            label_data = next(
                label_section["data"]
                for label_section in label_sections
                if label_section["experiment"] == accelerometer_file["experiment"]
            )
            sensor_data = np.hstack(
                (
                    zscore(
                        np.loadtxt(
                            fname=accelerometer_file["file_path"], dtype=np.float64
                        ),
                        axis=0,
                    ),
                    zscore(
                        np.loadtxt(fname=gyroscope_file["file_path"], dtype=np.float64),
                        axis=0,
                    ),
                )
            )
            # Create label sequence; some sections are not labeled, these stay labeled zero.
            labels = np.zeros(shape=(sensor_data.shape[0], 1), dtype=np.uint8)
            for label, start, end in label_data:
                labels[start - 1 : end] = label

            # Split the data into the datasets
            if accelerometer_file["user"] in training_user_numbers:
                data_train.append((sensor_data, labels))
            elif accelerometer_file["user"] in val_user_numbers:
                data_val.append((sensor_data, labels))
            elif accelerometer_file["user"] in test_user_numbers:
                data_test.append((sensor_data, labels))

        self._save_dataset(np.array(data_train, dtype=object), stype="train")
        self._save_dataset(np.array(data_val, dtype=object), stype="validation")
        self._save_dataset(np.array(data_test, dtype=object), stype="test")

    def _save_dataset(self, n, stype):
        X_temp = n[:, 0]
        y_temp = n[:, 1]
        dataframe = pd.DataFrame(X_temp[0])
        dataframe[6] = y_temp[0]
        for i in range(1, n.shape[0]):
            Xdf = pd.DataFrame(X_temp[i])
            Xdf[6] = y_temp[i]
            dataframe = dataframe.append(Xdf, ignore_index=True)

        dataframe.columns = [
            "accx",
            "accy",
            "accz",
            "gyrox",
            "gyroy",
            "gyroz",
            "labels",
        ]
        dataframe = dataframe.drop(dataframe[dataframe.labels == 0].index)

        path = stype + "_csv.csv"
        dataframe.to_csv(self.haptpath / path, sep=" ", encoding="utf-8", index=False)

    def _create_window(self, csvpath):
        df = pd.read_csv(csvpath, sep=" ")
        list_feat = []
        for i in range(0, int((len(df) - self.win_size) / self.win_shift) + 1):
            list_feat.append(
                df[i * self.win_shift : i * self.win_shift + self.win_size]
            )

        y_arr = np.array(list_feat, dtype=np.int32)[:, :, 6]

        y_lst = []
        for e in range(len(y_arr)):
            cn = np.bincount(y_arr[e])
            y_lst.append(np.argmax(cn))

        X = torch.tensor(np.array(list_feat)[:, :, :6], dtype=torch.float)
        y = torch.tensor(np.array(y_lst) - 1, dtype=torch.long)
        return X, y


@gin.configurable
class HAPTDataModule(pl.LightningDataModule):
    def __init__(self, run_paths, BATCH_SIZE, WIN_SHIFT, WIN_SIZE):
        super().__init__()
        self.batch_size = BATCH_SIZE
        self.win_shift = WIN_SHIFT
        self.win_size = WIN_SIZE
        self.num_workers = 0
        self.train_dataset = HAPTDataset(
            run_paths=run_paths,
            win_shift=self.win_shift,
            win_size=self.win_size,
            split="train",
        )
        self.validation_dataset = HAPTDataset(
            run_paths=run_paths,
            win_shift=self.win_shift,
            win_size=self.win_size,
            split="validation",
        )
        self.test_dataset = HAPTDataset(
            run_paths=run_paths,
            win_shift=self.win_shift,
            win_size=self.win_size,
            split="test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    @property
    def window_size(self):
        return self.win_size
