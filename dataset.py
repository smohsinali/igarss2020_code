import torch
import urllib.request
from tqdm import tqdm
import os
import datetime
import shapely
import json
import pandas as pd
import geopandas as gpd
import numpy as np

AUGMENTATION_OFFSET_MAGNITUDE = 1

class ModisDataset(torch.utils.data.Dataset):
    def __init__(self, region="africa",
                 fold="train",
                 verbose=True,
                 split_ratio = [.6,.2,.2],
                 seq_length=100,
                 overwrite=False,
                 include_time=False,
                 znormalize=False,
                 augment=False):
        super(ModisDataset).__init__()

        if region == "africa":
            self.dataset_url = "https://syncandshare.lrz.de/dl/fiQjtRdMiHJ9MC2a59LJ2wkc/africa_points.csv"
            self.dataset_local_path = "/tmp/africa_points.csv"
            self.dataset_local_npz = "/tmp/africa_points.npz"
            self.valuefield = "NDVI"
        elif region == "germany":
            self.dataset_url = "https://syncandshare.lrz.de/dl/fiSyp278PifiEghWNsnpH161/germany.csv"
            self.dataset_local_path = "/tmp/germany.csv"
            self.dataset_local_npz = "/tmp/germany.npz"
            self.valuefield = "NDVI"
        elif region == "canada":
            self.dataset_url = "https://syncandshare.lrz.de/dl/fiWPoesYqXuE1vUvMHR4EPzx/canada.csv"
            self.dataset_local_path = "/tmp/canada.csv"
            self.dataset_local_npz = "/tmp/canada.npz"
            self.valuefield = "EVI"
        elif region == "dubai":
            self.dataset_url = "https://syncandshare.lrz.de/dl/fiRtfxTyR5LWeG18HaNzg9Fz/dubai.csv"
            self.dataset_local_path = "/tmp/dubai.csv"
            self.dataset_local_npz = "/tmp/dubai.npz"
            self.valuefield = "EVI"

        assert sum(split_ratio) == 1

        np.random.seed(seed=0)
        self.fold = fold
        self.seq_length = seq_length
        self.split_ratio = split_ratio
        self.augment = augment

        self.verbose = verbose

        if not os.path.exists(self.dataset_local_path) or overwrite:
            self.print(f"No Dataset found at {self.dataset_local_path}. Downloading from {self.dataset_url}")
            download_url(self.dataset_url, self.dataset_local_path)
        else:
            self.print(f"local dataset found at {self.dataset_local_path}")

        if not os.path.exists(self.dataset_local_npz) or overwrite:
            self.print(f"no cached dataset found at {self.dataset_local_npz}. partitioning data in train/valid/test {split_ratio[0]}/{split_ratio[1]}/{split_ratio[2]} caching csv to npz files for faster loading")
            self.save_npz()

        self.print(f"loading cached dataset found at {self.dataset_local_npz}")
        self.data, self.meta = self.load_npz()


        data = self.data[:,:,1].astype(float)
        if znormalize:
            self.mean = np.nanmean(data)
            self.std = np.nanstd(data)
            data -= self.mean
            data /= self.std
        else:
            data *= 1e-4

        #ndvi = self.data[:,:,1].astype(float)  * 1e-4

        ndvi = interpolate_nans(data)

        self.date = self.data[:, :, 0]
        if include_time:
            dates = self.data[:, :, 0]
            #year = dates.astype('datetime64[Y]').astype(int) + 1970
            doy = dates.astype(np.datetime64) - dates.astype('datetime64[Y]')
            doy = doy.astype(float) / 365
            self.data = np.dstack([ndvi, doy])
        else:
            self.data = ndvi[:,:,None]

        self.x_data, self.y_data = transform_data(self.data, seq_len=self.seq_length)

    def load_npz(self):
        with np.load(self.dataset_local_npz, 'r', allow_pickle=True) as f:
            data = f[self.fold]
            meta = pd.DataFrame(f["meta"], columns=["fid","x","y","fold"])
        meta = meta.loc[meta["fold"]==self.fold].set_index("fid")
        return data, meta

    def save_npz(self):
        df = self._csv2dataframe()

        data, meta = self._dataframe2npz(df)

        meta = pd.DataFrame.from_records(meta)

        # shuffle meta dataframe
        meta = meta.sample(frac=1)

        N = len(meta)
        train, validate, test = np.split(meta, [int(self.split_ratio[0] * N),
                                                     int((self.split_ratio[0] + self.split_ratio[1]) * N)])

        train["fold"] = "train"
        validate["fold"] = "validate"
        test["fold"] = "test"
        meta = pd.concat([train, validate, test])

        store_dict = dict(
            train=data[train.fid],
            validate=data[validate.fid],
            test=data[test.fid],
            meta=meta
        )

        np.savez(self.dataset_local_npz, **store_dict)

    def _csv2dataframe(self):

        self.print(f"loading csv from {self.dataset_local_path}")
        df = gpd.read_file(self.dataset_local_path)

        self.print("convert string values to numeric")
        df[self.valuefield] = pd.to_numeric(df[self.valuefield], errors='coerce')

        self.print("write geometry object from string geojson")
        df["geometry"] = df[".geo"].apply(lambda x: shapely.geometry.shape(json.loads(x)))

        self.print("add feature id for each unique point")
        df["fid"], _ = pd.factorize(df[".geo"])
        df = df.set_index("fid")

        self.print("add date column from system:index")
        def systemindex2date(system_index):
            year, month, day, _ = system_index.split('_')
            return datetime.datetime(int(year), int(month), int(day))

        df["date"] = df["system:index"].apply(systemindex2date)

        return df

    def _dataframe2npz(self,df):
        data = list()
        meta = list()
        fids = df.index.unique()
        for fid in tqdm(fids, total=len(fids)):
            pt = df.loc[fid]
            date = pt["date"].dt.strftime("%Y-%m-%d").values
            value = pt[self.valuefield].values
            meta.append(dict(
                fid=fid,
                x=pt.iloc[0].geometry.x,
                y=pt.iloc[0].geometry.y
            ))
            pt = np.vstack([date, value]).T
            data.append(pt)
        return np.stack(data), np.stack(meta)

    def print(self,msg):
        if self.verbose:
            print(msg)

    def __len__(self):
        return self.x_data.shape[0]

    def add_offset(self, x, y):
        offset = (torch.rand(1)-0.5)*AUGMENTATION_OFFSET_MAGNITUDE
        x = x + offset
        y = y + offset
        return x,y

    def __getitem__(self,idx ):
        x = self.x_data[idx]
        y = self.y_data[idx]

        if self.augment:
            x,y = self.add_offset(x,y)

        return x, y


class Sentinel5Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 fold="train",
                 verbose=True,
                 split_ratio = [.6,.2,.2],
                 seq_length=100,
                 overwrite=False):
        super(Sentinel5Dataset).__init__()

        self.dataset_url = "https://syncandshare.lrz.de/dl/fiMBavGt2aPijY8pjiKESWFF/urbanareas_no2_all.csv"
        self.dataset_local_path = "/tmp/urbanareas_no2_all.csv"
        self.dataset_local_npz = "/tmp/urbanareas_no2_all.npz"

        assert sum(split_ratio) == 1

        np.random.seed(seed=0)
        self.fold = fold
        self.seq_length = seq_length
        self.split_ratio = split_ratio

        self.verbose = verbose

        if not os.path.exists(self.dataset_local_path) or overwrite:
            self.print(f"No Dataset found at {self.dataset_local_path}. Downloading from {self.dataset_url}")
            download_url(self.dataset_url, self.dataset_local_path)
        else:
            self.print(f"local dataset found at {self.dataset_local_path}")

        if not os.path.exists(self.dataset_local_npz) or overwrite:
            self.print(f"no cached dataset found at {self.dataset_local_npz}. partitioning data in train/valid/test {split_ratio[0]}/{split_ratio[1]}/{split_ratio[2]} caching csv to npz files for faster loading")
            self.save_npz()

        self.print(f"loading cached dataset found at {self.dataset_local_npz}")
        self.data, self.meta, self.mean, self.std = self.load_npz()


        x_data = list()
        y_data = list()
        for _,row in self.meta.iterrows():
            d = self.data[row["city"]][:, 1]
            x, y = transform_data(d.astype(float)[:, None, None], seq_len=self.seq_length)
            x_data.append(x)
            y_data.append(y)

        self.x_data = torch.cat(x_data,dim=0)
        self.y_data = torch.cat(y_data,dim=0)

        self.x_data = self.znormalize(self.x_data)
        self.y_data = self.znormalize(self.y_data)

    def znormalize(self,arr):
        arr = arr - torch.Tensor(self.mean)
        arr = arr / torch.Tensor(self.std)
        return arr

    def load_npz(self):
        with np.load(self.dataset_local_npz, 'r', allow_pickle=True) as f:
            meta = pd.DataFrame(f[self.fold], columns=["fid","x","y","city"])
            data = {k:v for k, v in f.items() if k in meta["city"].values}
            mean = f["mean"]
            std = f["std"]

        return data, meta, mean, std

    def save_npz(self):
        df = self._csv2dataframe()

        data_dict, meta = self._dataframe2npz(df)

        mean = df.value.mean()
        std = df.value.std()

        meta = pd.DataFrame.from_records(meta)

        # shuffle meta dataframe
        meta = meta.sample(frac=1)

        N = len(meta)
        train, validate, test = np.split(meta, [int(self.split_ratio[0] * N),
                                                     int((self.split_ratio[0] + self.split_ratio[1]) * N)])

        store_dict = data_dict
        store_dict["train"] = train
        store_dict["validate"] = validate
        store_dict["test"] = test
        store_dict["mean"] = mean
        store_dict["std"] = std

        np.savez(self.dataset_local_npz, **store_dict)

    def _csv2dataframe(self):

        self.print(f"loading csv from {self.dataset_local_path}")
        df = gpd.read_file(self.dataset_local_path)

        self.print("convert string values to numeric")
        df["value"] = pd.to_numeric(df["value"], errors='coerce')

        self.print("write geometry object from string geojson")
        df["geometry"] = df[".geo"].apply(lambda x: shapely.geometry.shape(json.loads(x)))

        self.print("add feature id for each unique point")
        df["fid"], _ = pd.factorize(df[".geo"])
        df = df.set_index("fid")

        df["date"] = pd.to_datetime(df["date"])

        return df

    def _dataframe2npz(self,df):
        fids = df.index.unique()
        data = dict()
        meta = list()
        for fid in tqdm(fids, total=len(fids)):
            pt = df.loc[fid]
            date = pt["date"].dt.strftime("%Y-%m-%d").values
            value = pt["value"].values
            city = pt["City"].iloc[0]
            meta.append(dict(
                fid=fid,
                x=pt.iloc[0].geometry.x,
                y=pt.iloc[0].geometry.y,
                city=city
            ))
            data[city]=np.vstack([date, value]).T
        return data, np.stack(meta)

    def print(self,msg):
        if self.verbose:
            print(msg)

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self,idx ):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y


def transform_data(arr, seq_len):
    d = arr.shape[2]
    x, y = [], []
    for i in range(len(arr) - seq_len):
        x_i = arr[i : i + seq_len]
        y_i = arr[i + 1 : i + seq_len + 1]
        x.append(x_i)
        y.append(y_i)
    x_arr = np.array(x).reshape(-1, seq_len, d)
    y_arr = np.array(y).reshape(-1, seq_len, d)
    x_var = torch.autograd.Variable(torch.from_numpy(x_arr).float())
    y_var = torch.autograd.Variable(torch.from_numpy(y_arr).float())
    return x_var, y_var

def interpolate_nans(arr):
    mask = np.isnan(arr)
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)