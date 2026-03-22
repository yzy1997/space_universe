import numpy as np
import pandas as pd


class MyPreprocessor:
    def __init__(self, base_columns,
                 lags_minutes=(5, 15, 30, 60),
                 roll_minutes=(30, 60),
                 add_physics_features=True):
        self.base_columns = list(base_columns)
        self.lags_minutes = tuple(lags_minutes)
        self.roll_minutes = tuple(roll_minutes)
        self.add_physics_features = bool(add_physics_features)

        self.feature_names_ = None
        self.step_minutes_ = None

    def _infer_step_minutes_from_index(self, dt_index: pd.DatetimeIndex) -> int:
        d = dt_index.to_series().diff().dropna()
        if len(d) == 0:
            return 1
        step = int(round(d.median().total_seconds() / 60.0))
        return max(step, 1)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 统一索引
        if "Date_UTC" in df.columns:
            df["Date_UTC"] = pd.to_datetime(df["Date_UTC"])
            df = df.sort_values("Date_UTC").set_index("Date_UTC")
        else:
            df = df.sort_index()

        # 只保留输入列
        missing = [c for c in self.base_columns if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in input df: {missing}")

        base = df[self.base_columns].copy().dropna()

        # 步长
        step = self._infer_step_minutes_from_index(base.index)
        self.step_minutes_ = step

        # ----------------------------
        # (1) 基础特征 + 物理派生特征
        # ----------------------------
        X0 = base.copy()

        if self.add_physics_features and set(["Bx","By","Bz"]).issubset(set(self.base_columns)):
            Bx, By, Bz = base["Bx"], base["By"], base["Bz"]

            X0["B_mag"] = np.sqrt(Bx**2 + By**2 + Bz**2)
            X0["B_xy"]  = np.sqrt(Bx**2 + By**2)
            X0["B_t"]   = np.sqrt(By**2 + Bz**2)
            X0["southward_Bz"] = np.maximum(0.0, -Bz)

            ca = np.arctan2(By.values, Bz.values)
            X0["clock_sin"] = np.sin(ca)
            X0["clock_cos"] = np.cos(ca)

        if self.add_physics_features and "sw_pressure" in self.base_columns:
            dP = base["sw_pressure"].diff()
            X0["dP"] = dP
            X0["abs_dP"] = dP.abs()

        # 后续 lag/rolling 只对这一份"基础特征集合"做
        base_feats = list(X0.columns)

        blocks = [X0]

        # ----------------------------
        # (2) lag 特征（只对 base_feats）
        # ----------------------------
        for lag_m in self.lags_minutes:
            p = int(round(lag_m / step))
            p = max(p, 1)
            X_lag = X0[base_feats].shift(p)
            X_lag.columns = [f"{c}_lag{lag_m}m" for c in base_feats]
            blocks.append(X_lag)

        # ----------------------------
        # (3) rolling 特征（只对 base_feats）
        # ----------------------------
        for win_m in self.roll_minutes:
            w = int(round(win_m / step))
            w = max(w, 2)

            rolled = X0[base_feats].rolling(window=w, min_periods=w)

            X_mean = rolled.mean()
            X_mean.columns = [f"{c}_rollmean{win_m}m" for c in base_feats]
            blocks.append(X_mean)

            X_std = rolled.std()
            X_std.columns = [f"{c}_rollstd{win_m}m" for c in base_feats]
            blocks.append(X_std)

        # 一次性拼接，避免碎片化
        X = pd.concat(blocks, axis=1).dropna()

        # 固化列顺序
        if self.feature_names_ is None:
            self.feature_names_ = list(X.columns)
        else:
            X = X.reindex(columns=self.feature_names_)

        return X