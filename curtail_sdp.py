"""
Hvenær á að skerða? Vatnsafl og forgangsorka (Verkefni 4, Aðgerðagreining 2).

Slembið kvikt bestunarlíkan (SDP) fyrir Fljótsdalsstöð:
  * gögn og mánaðarinnrennsli (hluti 1),
  * flokkun innrennslis og líkindafylki (hluti 2),
  * Bellman-lausn aftur á bak (hluti 3–4),
  * mat á stefnum: nákvæm upptalning allra innrennslisferla, dreifing á neti,
    keyrsla á sögulegum árum og klukkustundaendurspilun.

Allar einingar: rúmmál í Mm3 (= GL), afl í MW, rennsli í m3/s, líkanmánuður 730 klst.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------
# 1. Fastar úr verkefnislýsingu
# ----------------------------------------------------------------------------
K = 690 / 144             # MW á hvert m3/s  (P = K q)
Q_MAX = 144.0             # m3/s
P_MAX = 575.0             # MW (ein 115 MW vél í viðhaldi)
KLST = 730                # klst. í líkanmánuði
C = KLST * 3600 / 1e6     # 2,628 Mm3 á hvert m3/s yfir líkanmánuð
V_MIN, V_MAX = 100.0, 2270.0
FAST, AUKA = 483.0, 54.0  # MW forgangsorka og skerðanleg orka
A_VATN = C / K            # Mm3 sem 1 MW notar í einn líkanmánuð (0,5484)
M_JAFN = 1 / A_VATN       # MW-mán á Mm3: vatn í lok árs jafnverðmætt og afhending núna (1,8234)
V1_GILDI = [1600, 1800, 1950]
THJALFUNARAR = list(range(2015, 2024))
PROFUNARAR = 2024
MAN = ["jan", "feb", "mar", "apr", "maí", "jún", "júl", "ágú", "sep", "okt", "nóv", "des"]
FLOKKAR = ["Lítið", "Miðlungs", "Mikið"]
GWH_A_MWMAN = KLST / 1000  # 1 MW í einn líkanmánuð = 0,73 GWh

RADIR = {
    "H": "halslon_natturulegt_innrennsli.csv",       # m3/s, hrátt (EKKI slétta röðin)
    "J": "jokulsargong_rennsli.csv",                 # m3/s
    "yfirfall": "halslon_rennsli_um_yfirfall.csv",   # m3/s
    "notad_vatn": "fljotsdalsstod_notad_vatn.csv",   # m3/s
    "afl": "fljotsdalsstod_framleidsla.csv",         # MW
    "rummal": "halslon_rummal.csv",                  # GL = Mm3
}


# ----------------------------------------------------------------------------
# 2. Gögn (hluti 1)
# ----------------------------------------------------------------------------
def lesa_rod(mappa: str, skra: str) -> pd.DataFrame:
    """Les eina klukkustundarröð: Latin-1, ';', tugakomma, '---' = vantar."""
    d = pd.read_csv(os.path.join(mappa, skra), sep=";", encoding="latin-1", dtype=str)
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
    d = d.rename(columns={d.columns[0]: "dags", d.columns[1]: "klst", d.columns[2]: "txt"})
    d = d.dropna(subset=["dags"])
    d["timi"] = pd.to_datetime(d["dags"] + " " + d["klst"], format="%d.%m.%Y %H:%M:%S")
    txt = d["txt"].str.strip().replace({"---": None, "": None})
    d["gildi"] = pd.to_numeric(txt.str.replace(",", ".", regex=False), errors="coerce")
    flogg = [c for c in ("Valid", "Invalid", "Missing", "Quality", "Interpolation") if c in d.columns]
    return d[["timi", "gildi"] + flogg].reset_index(drop=True)


def lesa_gogn(mappa: str) -> dict:
    return {nafn: lesa_rod(mappa, skra) for nafn, skra in RADIR.items()}


def manadartafla(klst: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mánaðarrúmmál = meðaltal tiltækra klst.-gilda x klst. í almanaksmánuði x 0,0036 (Mm3)."""
    hlutar = []
    for nafn, d in klst.items():
        g = d.groupby([d["timi"].dt.year.rename("ar"), d["timi"].dt.month.rename("man")])["gildi"]
        hlutar.append(pd.DataFrame({f"{nafn}_medal": g.mean(), f"{nafn}_n": g.count()}))
    t = pd.concat(hlutar, axis=1)
    ar_, man_ = t.index.get_level_values("ar"), t.index.get_level_values("man")
    t["klst_i_man"] = pd.to_datetime(pd.DataFrame({"year": ar_, "month": man_, "day": 1})).dt.days_in_month.to_numpy() * 24
    for nafn in ("H", "J", "yfirfall", "notad_vatn"):
        t[f"{nafn}_Mm3"] = t[f"{nafn}_medal"] * t["klst_i_man"] * 0.0036
    t["innrennsli_Mm3"] = t["H_Mm3"] + t["J_Mm3"]
    t["framleidsla_GWh"] = t["afl_medal"] * t["klst_i_man"] / 1000
    t["nothaefur"] = (t["H_n"] >= 0.5 * t["klst_i_man"]) & (t["J_n"] >= 0.5 * t["klst_i_man"])
    t.loc[~t["nothaefur"], "innrennsli_Mm3"] = np.nan
    I = t["innrennsli_Mm3"].unstack("man")
    return t, I


# ----------------------------------------------------------------------------
# 3. Flokkun og líkindafylki (hluti 2)
# ----------------------------------------------------------------------------
def flokka(I_thj: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Mörk = 1/3 og 2/3 hlutfallsmörk hvers mánaðar (línuleg brúun); flokkur 0/1/2."""
    mork = np.array([np.quantile(I_thj[m].to_numpy(), [1 / 3, 2 / 3]) for m in range(1, 13)])
    flokkar = pd.DataFrame({m: np.searchsorted(mork[m - 1], I_thj[m].to_numpy(), side="right")
                            for m in range(1, 13)}, index=I_thj.index)
    return mork, flokkar


def flokkur(gildi, m: int, mork: np.ndarray):
    """Flokkur nýs gildis í almanaksmánuði m (1–12)."""
    return np.searchsorted(mork[m - 1], gildi, side="right")


def telja_umskipti(flokkar: pd.DataFrame) -> np.ndarray:
    """talningar[t, i, j]: fjöldi ára með flokk i í mánuði t-1 og flokk j í mánuði t (t=0: des árið á undan -> jan)."""
    n = np.zeros((12, 3, 3), dtype=int)
    ar = list(flokkar.index)
    for t in range(1, 12):
        for a in ar:
            n[t, flokkar.loc[a, t], flokkar.loc[a, t + 1]] += 1
    for a in ar:                       # des(a-1) -> jan(a), aðeins innan þjálfunaráranna
        if a - 1 in ar:
            n[0, flokkar.loc[a - 1, 12], flokkar.loc[a, 1]] += 1
    return n


def normal_flokkalikur(I_thj: pd.DataFrame, mork: np.ndarray, m: int) -> np.ndarray:
    """Flokkalíkur normaldreifingar sem er löguð að innrennsli mánaðar m (1–12) í þjálfunarárunum."""
    x = I_thj[m].to_numpy()
    mu, sd = x.mean(), x.std(ddof=1)
    F = stats.norm.cdf(mork[m - 1], mu, sd)
    return np.array([F[0], F[1] - F[0], 1 - F[1]])


def likindafylki(talningar: np.ndarray, I_thj: pd.DataFrame, mork: np.ndarray):
    """P[t, i, j] = Pr(C_t = j | C_{t-1} = i). Tóm röð -> normaldreifingarlíkur mánaðar t."""
    P = np.zeros((12, 3, 3))
    stadgenglar = []
    for t in range(12):
        for i in range(3):
            s = talningar[t, i].sum()
            if s > 0:
                P[t, i] = talningar[t, i] / s
            else:
                P[t, i] = normal_flokkalikur(I_thj, mork, t + 1)
                stadgenglar.append((t, i))
    return P, stadgenglar


def flokkamedaltol(I_thj: pd.DataFrame, flokkar: pd.DataFrame) -> np.ndarray:
    """Ibar[t, c] = meðalinnrennsli flokks c í mánuði t (Mm3)."""
    return np.array([[I_thj.loc[flokkar[t + 1] == c, t + 1].mean() for c in range(3)] for t in range(12)])


def des_dreifing(flokkar: pd.DataFrame) -> np.ndarray:
    """Reynsludreifing desemberflokks á þjálfunarárunum."""
    return np.bincount(flokkar[12].to_numpy(), minlength=3) / len(flokkar)


@dataclass
class Innrennslislikan:
    """Allt sem SDP-líkanið þarf um innrennsli."""
    mork: np.ndarray
    flokkar: pd.DataFrame
    talningar: np.ndarray
    P: np.ndarray
    Ibar: np.ndarray
    pi_des: np.ndarray
    stadgenglar: list = field(default_factory=list)


def smida_innrennslislikan(I_thj: pd.DataFrame) -> Innrennslislikan:
    mork, flokkar = flokka(I_thj)
    n = telja_umskipti(flokkar)
    P, st = likindafylki(n, I_thj, mork)
    return Innrennslislikan(mork, flokkar, n, P, flokkamedaltol(I_thj, flokkar), des_dreifing(flokkar), st)


# ----------------------------------------------------------------------------
# 4. Eðlisfræði eins mánaðar
# ----------------------------------------------------------------------------
def manudur(V, I):
    """Fyrir vatn í byrjun V og innrennsli I: (hámarksafl, afhent forgangsorka F, skortur D, hámark x).
    Forgangsorka fær vatnið fyrst; x <= 54 og x <= það sem eftir er af vatni og afli."""
    W = np.asarray(V, float) + np.asarray(I, float)
    B = np.maximum(0.0, W - V_MIN)
    Pbar = np.minimum(P_MAX, K * np.minimum(Q_MAX, B / C))
    F = np.minimum(FAST, Pbar)
    D = FAST - F
    xmax = np.clip(Pbar - F, 0.0, AUKA)
    return W, F, D, xmax


def lok_manadar(W, F, x):
    """Vatnsjafnvægi: V' = V + I - c q, yfirfall það sem fer yfir V_MAX."""
    Vp = W - C * (F + x) / K
    S = np.maximum(0.0, Vp - V_MAX)
    return np.minimum(Vp, V_MAX), S


# ----------------------------------------------------------------------------
# 5. SDP-líkanið
# ----------------------------------------------------------------------------
@dataclass
class Stillingar:
    V1: float = 1600.0          # upphafsstaða og markmið í lok árs
    M: float = 2 * M_JAFN       # refsing á hvern Mm3 sem vantar upp á V1 í lok árs (MW-mán/Mm3)
    lam: float = 0.0            # Lagrange-refsing fyrir ár með skorti á forgangsorku (MW-mán)
    mu: float = 10.0            # refsing á hvert MW-mán skorts á forgangsorku (forgangur líka milli mánaða)
    dV: float = 1.0             # netbil vatnsstöðu (Mm3)
    dx: float = 0.5             # netbil ákvörðunar (MW)
    jafntefli: float = 1e-3     # ef Q-gildi eru jöfn innan þessara marka (MW-mán) er stærra x valið


class SDP:
    """Bakvirk lausn Bellman-jöfnunnar á ástandinu (V, C, f):
    V = vatnsstaða í byrjun mánaðar, C = innrennslisflokkur mánaðarins (sést áður en ákveðið er),
    f = 1 ef skortur á forgangsorku hefur þegar orðið á árinu."""

    def __init__(self, il: Innrennslislikan, st: Stillingar, Ibar: np.ndarray | None = None):
        self.il, self.st = il, st
        self.Ibar = il.Ibar if Ibar is None else Ibar
        self.Vnet = np.round(np.arange(V_MIN, V_MAX + 1e-9, st.dV), 9)
        self.Xnet = np.round(np.arange(0.0, AUKA + 1e-9, st.dx), 9)
        nV = len(self.Vnet)
        self.J = np.zeros((12, 3, 2, nV))        # gildi eftir að innrennsli mánaðar sést
        self.G = np.zeros((12, 3, 2, nV))        # vænt framtíðargildi eftir ákvörðun (fall af V')
        self.stefna = np.zeros((12, 3, 2, nV))   # x*(t, c, f, V)

    # lokagildi: refsing fyrir að ná ekki V1 og Lagrange-refsing fyrir ár með skorti
    def lokagildi(self, V, f):
        return -self.st.M * np.maximum(0.0, self.st.V1 - V) - self.st.lam * f

    def _framhald(self, t, Vn, c, fn):
        """G_t(V', c, f') með línulegri brúun; c og fn eru fylki eða tölur."""
        out = np.empty_like(Vn)
        c = np.broadcast_to(c, Vn.shape[:1]); fn = np.broadcast_to(fn, Vn.shape[:1])
        for ci in range(3):
            for fi in range(2):
                r = (c == ci) & (fn == fi)
                if r.any():
                    out[r] = np.interp(Vn[r], self.Vnet, self.G[t, ci, fi])
        return out

    def akvarda(self, t, V, I, c, f, hluti=40000):
        """Besta x í mánuði t með einu Bellman-skrefi á raunverulegu ástandi (V, I, c, f).
        Skilar (x, Q-gildi, F, D). Ef jafnt er valið stærra x (afhent frekar en skert)."""
        V = np.atleast_1d(np.asarray(V, float)); I = np.broadcast_to(np.asarray(I, float), V.shape)
        c = np.broadcast_to(np.asarray(c), V.shape); f = np.broadcast_to(np.asarray(f), V.shape)
        if len(V) > hluti:     # skipt í búta til að spara minni
            bitar = [self.akvarda(t, V[i:i + hluti], I[i:i + hluti], c[i:i + hluti], f[i:i + hluti], hluti)
                     for i in range(0, len(V), hluti)]
            return tuple(np.concatenate([b[k] for b in bitar]) for k in range(4))
        W, F, D, xmax = manudur(V, I)
        fn = np.maximum(f, (D > 1e-9).astype(int))
        X = np.concatenate([np.broadcast_to(self.Xnet, (len(V), len(self.Xnet))), xmax[:, None]], axis=1)
        X = np.minimum(X, xmax[:, None])                         # óleyfileg gildi klippt að xmax
        Vn, _ = lok_manadar(W[:, None], F[:, None], X)
        n, a = X.shape
        cont = self._framhald(t, Vn.reshape(-1), np.repeat(c, a), np.repeat(fn, a)).reshape(n, a)
        Q = X - self.st.mu * D[:, None] + cont
        Qmax = Q.max(axis=1)
        valid = Q >= Qmax[:, None] - self.st.jafntefli
        x = np.where(valid, X, -1.0).max(axis=1)
        Qx = np.where(X == x[:, None], Q, -np.inf).max(axis=1)
        return x, Qx, F, D

    def leysa(self):
        nV = len(self.Vnet)
        for fi in range(2):
            self.G[11, :, fi] = self.lokagildi(self.Vnet, fi)
        for t in range(11, -1, -1):
            for ci in range(3):
                for fi in range(2):
                    x, Q, _, _ = self.akvarda(t, self.Vnet, self.Ibar[t, ci], np.full(nV, ci), np.full(nV, fi))
                    self.stefna[t, ci, fi], self.J[t, ci, fi] = x, Q
            if t > 0:
                # G_{t-1}(V', c_prev, f) = sum_c P_t(c_prev, c) J_t(V', c, f)
                self.G[t - 1] = np.einsum("ij,jfv->ifv", self.il.P[t], self.J[t])
        return self

    def U1(self, V, pi_des=None, f=0):
        """Vænt gildi í byrjun árs áður en janúarinnrennsli sést."""
        pi = self.il.pi_des if pi_des is None else np.asarray(pi_des, float)
        Jjan = np.array([np.interp(V, self.Vnet, self.J[0, c, f]) for c in range(3)])
        return float(pi @ self.il.P[0] @ Jjan)

    def stefnufall(self):
        return lambda t, V, I, c, f: self.akvarda(t, V, I, c, f)[0]


# ----------------------------------------------------------------------------
# 6. Samanburðarstefnur
# ----------------------------------------------------------------------------
def stefna_alltaf(t, V, I, c, f):
    """Afhenda 54 MW alltaf þegar vatn leyfir."""
    return manudur(V, I)[3]


def stefna_aldrei(t, V, I, c, f):
    """Aldrei afhenda skerðanlega orku."""
    return np.zeros_like(np.atleast_1d(np.asarray(V, float)))


class Vissujafngildi:
    """Vissujafngildisstefna (rúllandi áætlun með væntu innrennsli): í hverjum mánuði er leyst
    ákvarðanalaust vandamál þar sem framtíðarinnrennsli = skilyrt vænting miðað við núverandi flokk.
    Sama lokarefsing og sama skortsrefsing og í SDP (refsing lam á hvern skortsmánuð)."""

    def __init__(self, il: Innrennslislikan, st: Stillingar):
        self.il, self.st = il, st
        self.Vnet = np.round(np.arange(V_MIN, V_MAX + 1e-9, st.dV), 9)
        self.Xnet = np.round(np.arange(0.0, AUKA + 1e-9, st.dx), 9)
        self.H = np.zeros((12, 3, len(self.Vnet)))   # H[t, c](V'): gildi frá mánuði t+1 m.v. spá
        for t in range(12):
            for c in range(3):
                self.H[t, c] = self._akvardanalaust(t, c)

    def spa(self, t, c):
        """Vænt innrennsli mánaða t+1..11 miðað við C_t = c."""
        dreif = np.eye(3)[c]
        out = []
        for k in range(t + 1, 12):
            dreif = dreif @ self.il.P[k]
            out.append(float(dreif @ self.il.Ibar[k]))
        return out

    def _skref(self, Vnet, I, Hn):
        W, F, D, xmax = manudur(Vnet, I)
        X = np.minimum(np.concatenate([np.broadcast_to(self.Xnet, (len(Vnet), len(self.Xnet))), xmax[:, None]], 1), xmax[:, None])
        Vn, _ = lok_manadar(W[:, None], F[:, None], X)
        Q = X - self.st.mu * D[:, None] - self.st.lam * (D[:, None] > 1e-9) + np.interp(Vn, self.Vnet, Hn)
        return Q.max(1)

    def _akvardanalaust(self, t, c):
        Hn = -self.st.M * np.maximum(0.0, self.st.V1 - self.Vnet)
        for I in reversed(self.spa(t, c)):
            Hn = self._skref(self.Vnet, I, Hn)
        return Hn

    def __call__(self, t, V, I, c, f, hluti=40000):
        V = np.atleast_1d(np.asarray(V, float)); I = np.broadcast_to(np.asarray(I, float), V.shape)
        c = np.broadcast_to(np.asarray(c), V.shape)
        if len(V) > hluti:
            return np.concatenate([self(t, V[i:i + hluti], I[i:i + hluti], c[i:i + hluti], None, hluti)
                                   for i in range(0, len(V), hluti)])
        W, F, D, xmax = manudur(V, I)
        X = np.minimum(np.concatenate([np.broadcast_to(self.Xnet, (len(V), len(self.Xnet))), xmax[:, None]], 1), xmax[:, None])
        Vn, _ = lok_manadar(W[:, None], F[:, None], X)
        cont = np.empty_like(Vn)
        for ci in range(3):
            r = c == ci
            if r.any():
                cont[r] = np.interp(Vn[r], self.Vnet, self.H[t, ci])
        Q = X + cont
        Qmax = Q.max(1)
        return np.where(Q >= Qmax[:, None] - self.st.jafntefli, X, -1.0).max(1)


# ----------------------------------------------------------------------------
# 7. Mat: nákvæm upptalning allra flokkaferla líkansins
# ----------------------------------------------------------------------------
VIK_MARKMID = 1.0   # Mm3: lokastaða telst ná markmiði ef V13 >= V1 - 1 (námundun ákvarðananets)


def meta_slodir(stefna, il: Innrennslislikan, V1: float, pi_des=None, Ibar=None, P=None, geyma=False):
    """Telur upp alla flokkaferla (des, jan, ..., des) með jákvæðar líkur og keyrir stefnuna á þá.
    Innrennsli = flokkameðaltal. Skilar dict með væntigildum, mánaðarsniðum og (ef geyma) ferlum."""
    Ibar = il.Ibar if Ibar is None else Ibar
    P = il.P if P is None else P
    pi = il.pi_des if pi_des is None else np.asarray(pi_des, float)
    # mánuður 0: des-flokkur d -> jan-flokkur c
    d0 = np.repeat(np.arange(3), 3); c0 = np.tile(np.arange(3), 3)
    p = pi[d0] * P[0][d0, c0]
    keep = p > 0
    p, c = p[keep], c0[keep]
    n = len(p)
    V = np.full(n, float(V1)); f = np.zeros(n, int)
    sx = np.zeros(n); sD = np.zeros(n); sS = np.zeros(n); nD = np.zeros(n); Vmin = V.copy()
    saga = {"x": [], "V": [], "c": []} if geyma else None
    man = {k: np.zeros(12) for k in ("Ex", "Pskert", "ED", "ES", "EV", "Pskortur", "EI")}
    for t in range(12):
        I = Ibar[t, c]
        x = stefna(t, V, I, c, f)
        W, F, D, xmax = manudur(V, I)
        x = np.clip(x, 0.0, xmax)
        Vn, S = lok_manadar(W, F, x)
        man["Ex"][t] = p @ x; man["Pskert"][t] = p @ (x < AUKA - 1e-6); man["ED"][t] = p @ D
        man["ES"][t] = p @ S; man["EV"][t] = p @ Vn; man["Pskortur"][t] = p @ (D > 1e-9); man["EI"][t] = p @ I
        sx += x; sD += D; sS += S; nD += D > 1e-9
        f = np.maximum(f, (D > 1e-9).astype(int))
        V = Vn; Vmin = np.minimum(Vmin, V)
        if geyma:
            saga["x"].append(x); saga["V"].append(V); saga["c"].append(c)
        if t < 11:   # greinum í næsta flokk
            pr = P[t + 1][c]                       # (n, 3)
            idx, cn = np.nonzero(pr > 0)
            p = p[idx] * pr[idx, cn]; c = cn
            V, f, sx, sD, sS, nD, Vmin = V[idx], f[idx], sx[idx], sD[idx], sS[idx], nD[idx], Vmin[idx]
            if geyma:
                for k in saga:
                    saga[k] = [a[idx] for a in saga[k]]
    halli = np.maximum(0.0, V1 - V)
    res = {
        "p": p, "V13": V, "sum_x": sx, "sum_D": sD, "sum_S": sS, "f": f, "Vmin": Vmin, "halli": halli,
        "E_afhent_GWh": p @ sx * GWH_A_MWMAN,
        "E_skert_GWh": (12 * AUKA - p @ sx) * GWH_A_MWMAN,
        "areidanleiki": 1 - p @ f,
        "E_skortur_GWh": p @ sD * GWH_A_MWMAN,
        "E_yfirfall": p @ sS,
        "E_V13": p @ V,
        "P_V13_ge_V1": p @ (V >= V1 - VIK_MARKMID),
        "E_halli": p @ halli,
        "fjoldi_ferla": len(p),
        "man": man,
    }
    if geyma:
        res["saga"] = {k: np.array(v).T for k, v in saga.items()}
    return res


def meta_net(stefnutafla, il: Innrennslislikan, V1: float, Vnet, Ibar=None, P=None, pi_des=None):
    """Dreifir líkindamassa á netinu (V, c, f) með línulegri skiptingu milli netpunkta.
    stefnutafla[t, c, f, iV] = x á netinu. Hraðvirkt mat sem nota má innan leitar og óvissugreiningar."""
    Ibar = il.Ibar if Ibar is None else Ibar
    P = il.P if P is None else P
    pi = il.pi_des if pi_des is None else np.asarray(pi_des, float)
    nV, dV = len(Vnet), Vnet[1] - Vnet[0]
    massi = np.zeros((3, 2, nV))
    i1 = int(round((V1 - V_MIN) / dV))
    massi[:, 0, i1] = pi @ P[0]
    Ex = ED = ES = 0.0
    for t in range(12):
        nyr = np.zeros((3, 2, nV))
        for c in range(3):
            W, F, D, xmax = manudur(Vnet, Ibar[t, c])
            fn_skortur = (D > 1e-9).astype(int)
            for f in range(2):
                m = massi[c, f]
                if m.sum() <= 0:
                    continue
                x = np.minimum(stefnutafla[t, c, f], xmax)
                Vn, S = lok_manadar(W, F, x)
                Ex += m @ x; ED += m @ D; ES += m @ S
                pos = (Vn - V_MIN) / dV
                i0 = np.clip(np.floor(pos).astype(int), 0, nV - 1)
                w = np.where(i0 < nV - 1, pos - i0, 0.0)
                skipt = np.zeros((2, nV))
                fn = np.maximum(f, fn_skortur)
                for fi in range(2):
                    r = fn == fi
                    if r.any():
                        np.add.at(skipt[fi], i0[r], m[r] * (1 - w[r]))
                        np.add.at(skipt[fi], np.minimum(i0[r] + 1, nV - 1), m[r] * w[r])
                if t < 11:
                    nyr += P[t + 1][c][:, None, None] * skipt[None]
                else:
                    nyr[c] += skipt
        massi = nyr
    lok = massi.sum(axis=0)                      # (2, nV) eftir des
    halli = np.maximum(0.0, V1 - Vnet)
    return {
        "E_afhent_GWh": Ex * GWH_A_MWMAN, "E_skert_GWh": (12 * AUKA - Ex) * GWH_A_MWMAN,
        "E_skortur_GWh": ED * GWH_A_MWMAN, "E_yfirfall": ES,
        "areidanleiki": lok[0].sum(), "E_V13": lok.sum(0) @ Vnet,
        "P_V13_ge_V1": lok.sum(0) @ (Vnet >= V1 - VIK_MARKMID), "E_halli": lok.sum(0) @ halli,
    }


def finna_lam(il: Innrennslislikan, st: Stillingar, krafa=0.95, tol=1e-3, matsfall="slodir"):
    """Ytri leit (helmingun) að minnsta lam þannig að Pr(ár án skorts) >= krafa.
    Skilar (lam, SDP, mat, stada) þar sem stada er 'obindandi', 'bindandi' eða 'engin_stefna'."""
    def leysa(lam):
        s = Stillingar(**{**st.__dict__, "lam": lam})
        sdp = SDP(il, s).leysa()
        r = meta_slodir(sdp.stefnufall(), il, s.V1) if matsfall == "slodir" else meta_net(sdp.stefna, il, s.V1, sdp.Vnet)
        return sdp, r
    sdp, r = leysa(0.0)
    if r["areidanleiki"] >= krafa:
        return 0.0, sdp, r, "obindandi"
    r_aldrei = meta_slodir(stefna_aldrei, il, st.V1)
    if r_aldrei["areidanleiki"] < krafa:          # aldrei-afhenda er áreiðanlegasta stefnan
        return np.nan, None, r_aldrei, "engin_stefna"
    lo, hi = 0.0, 1.0
    while leysa(hi)[1]["areidanleiki"] < krafa:
        hi *= 4.0
    while hi - lo > tol * max(1.0, hi):
        mid = 0.5 * (lo + hi)
        if leysa(mid)[1]["areidanleiki"] >= krafa:
            hi = mid
        else:
            lo = mid
    sdp, r = leysa(hi)
    return hi, sdp, r, "bindandi"


def fullkomin_vitneskja(Ipaths, V1: float):
    """Efri mörk: besta ákvörðun þegar allt innrennsli ársins er þekkt fyrir fram (gildir þegar forgangsorka
    er aldrei í hættu). Fyrst er allt afhent; síðan er skert í síðustu mánuðum fyrst, aðeins það vatn sem
    skilar sér í lok árs (veldur ekki yfirfalli síðar), uns V13 = V1 eða ekki er hægt að gera betur.
    Rétt fyrir hvaða M > M_JAFN sem er."""
    n = Ipaths.shape[0]; Y = A_VATN * AUKA; firm = A_VATN * FAST
    V = np.full(n, float(V1)); Vend = np.zeros((n, 12))
    for t in range(12):
        Vp = V + Ipaths[:, t] - firm - Y
        if (Vp < V_MIN - 1e-9).any():
            raise ValueError("Forgangsorka í hættu: fullkomin vitneskja með einföldu reikniriti á ekki við")
        V = np.minimum(Vp, V_MAX); Vend[:, t] = V
    halli = np.maximum(0.0, V1 - Vend[:, 11]); y = np.zeros((n, 12))
    for t in range(11, -1, -1):
        rymi = (V_MAX - Vend[:, t:]).min(axis=1)
        yt = np.minimum(np.minimum(Y, halli), rymi)
        y[:, t] = yt; halli -= yt; Vend[:, t:] += yt[:, None]
    return AUKA - y / A_VATN, Vend, halli


def keyra_ar(stefna, I_ar, c_ar, V1: float):
    """Keyrir stefnu á eitt ár af raunverulegu innrennsli (12 gildi) og flokkum þess."""
    V, f = float(V1), 0
    lin = []
    for t in range(12):
        x = float(stefna(t, np.array([V]), np.array([I_ar[t]]), np.array([c_ar[t]]), np.array([f]))[0])
        W, F, D, xmax = manudur(V, I_ar[t])
        x = float(np.clip(x, 0.0, xmax))
        Vn, S = lok_manadar(W, F, x)
        lin.append(dict(man=MAN[t], I=I_ar[t], flokkur=FLOKKAR[int(c_ar[t])], V_byrjun=V, x=x, F=float(F),
                        D=float(D), yfirfall=float(S), V_lok=float(Vn)))
        f = max(f, int(D > 1e-9)); V = float(Vn)
    return pd.DataFrame(lin)


def samantekt_ars(df: pd.DataFrame, V1: float) -> dict:
    return {
        "afhent_GWh": df["x"].sum() * GWH_A_MWMAN,
        "skert_GWh": (12 * AUKA - df["x"].sum()) * GWH_A_MWMAN,
        "skertir_man": int((df["x"] < AUKA - 1e-6).sum()),
        "skortur_GWh": df["D"].sum() * GWH_A_MWMAN,
        "skortsmanudir": int((df["D"] > 1e-9).sum()),
        "yfirfall": df["yfirfall"].sum(),
        "V13": df["V_lok"].iloc[-1],
        "Vmin": df["V_lok"].min(),
        "halli": max(0.0, V1 - df["V_lok"].iloc[-1]),
    }
# ----------------------------------------------------------------------------
# 8. Tengingar við hluta 2 í aðal-Colab (engin endurútreikningur á líkindafylkjum)
# ----------------------------------------------------------------------------
def innrennslislikan_ur_hluta2(mork, flokkar, talningar, P_man, I_flokkar):
    """Býr til inntak fyrir SDP úr föstum niðurstöðum hluta 2.

    mork: DataFrame með index 1/3, 2/3 og dálka 1..12.
    flokkar: DataFrame með árum 2015..2023 og mánuðum 1..12.
    talningar, P_man: dict lyklað eftir mánuðum 1..12.
    I_flokkar: 12 x 3 fylki (Mm³).

    Nota beint eftir hluti_2_4():
       il = cs.innrennslislikan_ur_hluta2(mork, flokkar, talningar, P_man, I_flokkar)
    """
    markafylki = np.array([[mork.loc[1 / 3, man], mork.loc[2 / 3, man]]
                           for man in range(1, 13)], dtype=float)
    n = np.stack([np.asarray(talningar[man], dtype=int) for man in range(1, 13)])
    P = np.stack([np.asarray(P_man[man], dtype=float) for man in range(1, 13)])
    Ibar = np.asarray(I_flokkar, dtype=float)
    if markafylki.shape != (12, 2) or n.shape != (12, 3, 3) or P.shape != (12, 3, 3) or Ibar.shape != (12, 3):
        raise ValueError("Röng stærð á gögnum úr hluta 2.")
    if not np.isfinite(P).all() or not np.allclose(P.sum(axis=2), 1):
        raise ValueError("Umskiptalíkindin úr hluta 2 verða að vera endanleg og raðsamtala 1.")
    if not np.isfinite(Ibar).all():
        raise ValueError("Meðalinnrennsli þarf að vera skilgreint fyrir alla flokka.")
    pi_des = des_dreifing(flokkar)
    stadgenglar = [(man - 1, i) for man in range(1, 13) for i in range(3)
                   if np.asarray(talningar[man])[i].sum() == 0]
    return Innrennslislikan(markafylki, flokkar, n, P, Ibar, pi_des, stadgenglar)

# ----------------------------------------------------------------------------
# Myndir 3–4 fyrir sameinaða Colab-skýrslu
# ----------------------------------------------------------------------------
def _mynd_gogn(grunnur):
    """Reiknar vægt miðgildi og 5/95% mörk úr nákvæmum innrennslisferlum."""
    gogn = {}
    for V1, g in grunnur.items():
        rs = g['rs']
        if 'saga' not in rs:
            raise ValueError('Keyrðu meta_slodir(..., geyma=True) áður en myndir eru teiknaðar.')
        p = np.asarray(rs['p'])
        V = np.asarray(rs['saga']['V'])
        def hlutf(a, q):
            o = np.argsort(a)
            return a[o][np.searchsorted(np.cumsum(p[o]), q, side='left')]
        gogn[V1] = {
            'Ex': np.asarray(rs['man']['Ex']),
            'V_hlutf': {q: np.array([hlutf(V[:, t], q) for t in range(12)])
                        for q in (0.05, 0.5, 0.95)},
        }
    return gogn


def teikna_stefnukort(grunnur, vista_i=None):
    """Mynd 3 úr SDP-Notebook: hitakort afhendingarstefnu fyrir V1=1600 og 1950."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    snid = _mynd_gogn(grunnur)
    man = np.arange(1, 13)
    man_stafir = [m[0].upper() for m in MAN]
    cmap = LinearSegmentedColormap.from_list(
        'blar', ['#f6f5f1', '#6da7ec', '#3987e5', '#2a78d6', '#1c5cab', '#104281', '#0d366b']
    )
    fig, axs = plt.subplots(2, 3, figsize=(8.6, 5.6), sharex=True, sharey=True)
    for row, V1 in enumerate((1600, 1950)):
        sdp = grunnur[V1]['sdp']
        V_byrjun_mid = np.r_[V1, snid[V1]['V_hlutf'][0.5][:11]]
        for c in range(3):
            ax = axs[row, c]
            im = ax.imshow(sdp.stefna[:, c, 0].T, origin='lower', aspect='auto',
                           cmap=cmap, vmin=0, vmax=54,
                           extent=[0.5, 12.5, V_MIN, V_MAX], interpolation='nearest')
            ax.grid(False)
            ax.plot(man, V_byrjun_mid, color='#eb6834', lw=1.6,
                    label='Miðgildi vatnsstöðu')
            ax.axhline(V1, color='#0b0b0b', lw=0.8, ls=':')
            ax.set_xticks(man, man_stafir, fontsize=7.5)
            if row == 0:
                ax.set_title(f'{FLOKKAR[c]} innrennsli', fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f'V₁ = {V1}\nvatn í byrjun mán. (Mm³)', fontsize=8.5)
    axs[0, 0].legend(loc='lower left', fontsize=7)
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.02)
    cb.set_label('Afhent viðbótarorka x* (MW)')
    fig.suptitle('Mynd 3. SDP-stefnan (ljóst = skert, dökkt = 54 MW afhent; punktalína = V₁)',
                 fontsize=10.5, fontweight='bold', x=0.02, ha='left')
    if vista_i:
        fig.savefig(vista_i, bbox_inches='tight', dpi=200)
    plt.show()
    return fig


def teikna_ferla_og_skerdingu(grunnur, vista_i=None):
    """Mynd 4 úr SDP-Notebook: dreifing vatnsstöðu og vænt skerðing."""
    import matplotlib.pyplot as plt
    snid = _mynd_gogn(grunnur)
    man = np.arange(1, 13)
    man_stafir = [m[0].upper() for m in MAN]
    litir = dict(zip(V1_GILDI, ('#2a78d6', '#eb6834', '#1baf7a')))
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.4))
    ax = axs[0]
    for V1 in V1_GILDI:
        h = snid[V1]['V_hlutf']
        ax.fill_between(man, h[0.05], h[0.95], color=litir[V1], alpha=0.12, lw=0)
        ax.plot(man, h[0.5], color=litir[V1], label=f'V₁ = {V1}')
    ax.axhline(V_MAX, color='#52514e', lw=0.8, ls='--')
    ax.text(1, V_MAX - 110, 'V_max = 2270', fontsize=7.5, color='#52514e')
    ax.axhline(V_MIN, color='#d03b3b', lw=0.8, ls='--')
    ax.text(1, V_MIN + 30, 'Varaforði 100', fontsize=7.5, color='#d03b3b')
    ax.set_xticks(man, man_stafir)
    ax.set_ylabel('Vatnsstaða í lok mánaðar (Mm³)')
    ax.set_title('a) Miðgildi og 5–95% bil')
    ax.legend(fontsize=8, loc='lower right')
    ax = axs[1]
    for k, V1 in enumerate(V1_GILDI):
        ax.bar(man + (k - 1) * 0.27, AUKA - snid[V1]['Ex'], width=0.24,
               color=litir[V1], label=f'V₁ = {V1}')
    ax.set_xticks(man, man_stafir)
    ax.set_ylabel('Vænt skerðing 54 − E[x] (MW)')
    ax.set_title('b) Vænt skerðing eftir mánuðum')
    ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('Mynd 4. SDP-stefnan á öllum innrennslisferlum líkansins',
                 fontsize=10.5, fontweight='bold', x=0.02, ha='left')
    fig.tight_layout()
    if vista_i:
        fig.savefig(vista_i, bbox_inches='tight', dpi=200)
    plt.show()
    return fig

# ----------------------------------------------------------------------------
# 9. Samanburður afhendingarstefna (úr SDP-Notebook, kafli 4)
# ----------------------------------------------------------------------------
def samanburdur_stefna(grunnur, il):
    """Ber saman SDP við einfaldari stefnur og fullkomna vitneskju.

    Notar NÁKVÆMLEGA sömu líkur og innrennslisferla og meta_slodir.
    grunnur[V1]['rs'] þarf að innihalda 'saga' (geyma=True).

    Skilar (samanburdur, likindatafla, skerding_tafla, snid).
    Þessi föll prenta/teikna ekki sjálf; Colab ræður framsetningunni.
    """
    def r(x, n):
        return round(float(x), n)

    def malikvardar(res, V1, nafn, M):
        out = {
            'stefna': nafn, 'V1': V1,
            'tekjur_%': r(100 * res['E_afhent_GWh'] / (12 * AUKA * GWH_A_MWMAN), 1),
            'afhent_GWh': r(res['E_afhent_GWh'], 1),
            'skert_GWh': r(res['E_skert_GWh'], 1),
            'áreiðanleiki': r(res['areidanleiki'], 4),
            'skortur_GWh': r(res['E_skortur_GWh'], 2),
            'yfirfall_Mm3': r(res['E_yfirfall'], 0),
            'E_V13': r(res['E_V13'], 0),
            'P(V13≥V1)': r(res['P_V13_ge_V1'], 3),
            'E_halli_Mm3': r(res['E_halli'], 1),
        }
        if 'sum_x' in res:
            out['markgildi'] = r(res['p'] @ (res['sum_x'] - M * res['halli']), 1)
        return out

    samanb, snid = [], {}
    for V1, g in grunnur.items():
        st, rs, p = g['st'], g['rs'], g['rs']['p']
        if 'saga' not in rs:
            raise ValueError(f'V1={V1}: keyra þarf meta_slodir(..., geyma=True)')
        ce = Vissujafngildi(il, st)
        res = {
            'Alltaf afhenda': meta_slodir(stefna_alltaf, il, V1),
            'Aldrei afhenda': meta_slodir(stefna_aldrei, il, V1),
            'Vissujafngildi': meta_slodir(ce, il, V1),
            'SDP': rs,
        }
        for nafn, rr in res.items():
            samanb.append(malikvardar(rr, V1, nafn, st.M))

        # Fullkomin vitneskja: efri mörk með framtíðarinnrennsli þekkt fyrirfram.
        Ip = np.array([il.Ibar[t, rs['saga']['c'][:, t]] for t in range(12)]).T
        xpi, Vend, hpi = fullkomin_vitneskja(Ip, V1)
        Vb = np.concatenate([np.full((len(p), 1), float(V1)), Vend[:, :11]], axis=1)
        pi = {
            'E_afhent_GWh': p @ xpi.sum(1) * GWH_A_MWMAN,
            'E_skert_GWh': (12 * AUKA - p @ xpi.sum(1)) * GWH_A_MWMAN,
            'areidanleiki': 1.0,
            'E_skortur_GWh': 0.0,
            'E_V13': p @ Vend[:, 11],
            'E_halli': p @ hpi,
            'P_V13_ge_V1': p @ (Vend[:, 11] >= V1 - VIK_MARKMID),
            'E_yfirfall': p @ np.maximum(
                0, Vb + Ip - A_VATN * (FAST + xpi) - V_MAX
            ).sum(1),
        }
        m = malikvardar(pi, V1, 'Fullkomin vitneskja', st.M)
        m['markgildi'] = r(p @ (xpi.sum(1) - st.M * hpi), 1)
        samanb.append(m)

        Vs, xs = rs['saga']['V'], rs['saga']['x']

        def hlutf(a, q):
            o = np.argsort(a)
            return a[o][np.searchsorted(np.cumsum(p[o]), q)]

        snid[V1] = {
            'Ex': rs['man']['Ex'],
            'Ex_CE': res['Vissujafngildi']['man']['Ex'],
            'Ex_PI': p @ xpi,
            'Pskert': rs['man']['Pskert'],
            'ES': rs['man']['ES'],
            'V_hlutf': {
                q: np.array([hlutf(Vs[:, t], q) for t in range(12)])
                for q in (0.05, 0.25, 0.5, 0.75, 0.95)
            },
            'P_aldrei_skert': r(p @ (rs['sum_x'] >= 12 * AUKA - 1e-6), 3),
            'P_skert_jan_jul': r(p @ (xs[:, :7] < AUKA - 1e-6).any(1), 3),
            'P_skert_agu_des': r(p @ (xs[:, 7:] < AUKA - 1e-6).any(1), 3),
            'P_fullt_sep': r(p @ (Vs[:, 8] >= V_MAX - 1e-6), 3),
            'P_yfirfall': r(p @ (rs['sum_S'] > 1e-6), 3),
        }

    samanb = pd.DataFrame(samanb)
    likindatafla = pd.DataFrame({
        V1: {k: v for k, v in s.items() if k.startswith('P_')}
        for V1, s in snid.items()
    })
    skerding_tafla = pd.DataFrame({
        V1: np.round(AUKA - s['Ex'], 2) for V1, s in snid.items()
    }, index=MAN).T.rename_axis('Vænt skerðing (MW)')
    return samanb, likindatafla, skerding_tafla, snid


# ----------------------------------------------------------------------------
# 11. Söguleg ár og prófunarár: samantekt SDP-stefnu (myndræn mánaðarröð)
# ----------------------------------------------------------------------------
def soguleg_ar_tafla(grunnur, il, I, ar_fra=2015, ar_til=2024):
    """Endurskapar töfluna úr SDP-Notebookinu yfir árlegar keyrslur.

    Tekur við upphafsstöðunum í ``grunnur`` og mánaðarlegu innrennsli
    ``I`` úr hluta 1 (DataFrame, index=ár, dálkar 1..12).

    Skilar (soguleg, tafla), þar sem tafla hefur dálka fyrir skerðingu,
    mánaðarmynstur, lokavatnsstöðu og yfirfall. Einnig eru varðveittar
    niðurstöður annarra tveggja samanburðarstefna í soguleg.

    Tákn: █ full 54 MW afhending, ▒ skerðing að hluta, · full skerðing.
    Ár 2024 er prófunarár utan þjálfunargagna.
    """
    ar = list(range(ar_fra, ar_til + 1))
    vantar = [a for a in ar if a not in I.index]
    if vantar:
        raise ValueError(f"Innrennslisgögn vantar fyrir árin: {vantar}")

    sogu = []
    for V1, g in grunnur.items():
        stefnur = {
            'Alltaf afhenda': stefna_alltaf,
            'Vissujafngildi': Vissujafngildi(il, g['st']),
            'SDP': g['sdp'].stefnufall(),
        }
        for a in ar:
            I_ar = I.loc[a, list(range(1, 13))].to_numpy(dtype=float)
            c_ar = np.array([flokkur(I_ar[m], m + 1, il.mork) for m in range(12)])
            for nafn, stefna in stefnur.items():
                df = keyra_ar(stefna, I_ar, c_ar, V1)
                sam = {k: round(float(v), 1) for k, v in samantekt_ars(df, V1).items()}
                munstur = ''.join(
                    '·' if x < 0.01 else ('▒' if x < AUKA - 1e-6 else '█')
                    for x in df['x']
                )
                sogu.append(dict(V1=V1, ar=a, stefna=nafn, **sam, manudir=munstur))

    soguleg = pd.DataFrame(sogu)
    tafla = soguleg.loc[soguleg['stefna'] == 'SDP'].pivot(
        index='ar', columns='V1', values=['skert_GWh', 'manudir', 'V13', 'yfirfall']
    )
    return soguleg, tafla

# ============================================================================
# VIÐBÓT: Allar eftirfarandi myndir og töflur úr upprunalegu SDP-Notebooki.
# Kallað á föllin úr Colab; þau breyta hvorki il né grunnur.
# ============================================================================

def _litasnid():
    """Sameiginlegt litaval og útlit SDP-skýrslunnar."""
    lit = dict(blar='#2a78d6', app='#eb6834', gr='#1baf7a', gulur='#eda100',
               blek2='#52514e', hraedd='#d03b3b', daufur='#898781')
    return lit, dict(zip(V1_GILDI, (lit['blar'], lit['app'], lit['gr'])))


def _birta_mynd(fig, vista_i=None):
    import matplotlib.pyplot as plt
    if vista_i is not None:
        fig.savefig(vista_i, bbox_inches='tight', dpi=200)
    plt.show()
    return fig


def naemni_lokamarkmids(il, margfeldi=(1.01, 1.25, 1.5, 2, 3, 5, 10, 30, 100),
                        teikna=True, vista_i=None):
    """Mynd 5: næmni SDP-stefnunnar gagnvart lokarefsingu M.

    Endurleysir Bellman fyrir hvert M og V1. Skilar (tafla, pivot).
    """
    import matplotlib.pyplot as plt
    linur = []
    for V1 in V1_GILDI:
        for marg in margfeldi:
            st = Stillingar(V1=V1, M=float(marg) * M_JAFN)
            sd = SDP(il, st).leysa()
            rr = meta_slodir(sd.stefnufall(), il, V1)
            linur.append(dict(
                V1=V1, M_sinnum_Mstjarna=float(marg),
                skert_GWh=round(float(rr['E_skert_GWh']), 1),
                P_V13_ge_V1=round(float(rr['P_V13_ge_V1']), 3),
                E_halli=round(float(rr['E_halli']), 2),
                skert_jan_jul_MWman=round(float((AUKA - rr['man']['Ex'][:7]).sum()), 1)))
    tafla = pd.DataFrame(linur)
    pivot = tafla.pivot(index='M_sinnum_Mstjarna', columns='V1',
                        values=['skert_GWh', 'P_V13_ge_V1', 'skert_jan_jul_MWman'])
    if teikna:
        lit, litir = _litasnid()
        fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
        for V1 in V1_GILDI:
            d = tafla[tafla.V1 == V1]
            axs[0].plot(d.M_sinnum_Mstjarna, d.skert_GWh,
                        color=litir[V1], marker='o', ms=4, label=f'V₁ = {V1}')
            axs[1].plot(d.M_sinnum_Mstjarna, 100 * d.P_V13_ge_V1,
                        color=litir[V1], marker='o', ms=4, label=f'V₁ = {V1}')
        for ax in axs:
            ax.set_xscale('log')
            ax.set_xlabel('M / M*   (M* = 1,82 MW-mán/Mm³)')
            ax.axvline(2, color=lit['blek2'], lw=0.8, ls=':')
        axs[0].set_ylabel('Vænt skert orka (GWh/ár)')
        axs[0].set_title('a) Kostnaður lokamarkmiðs')
        axs[1].set_ylabel('Pr(V₁₃ ≥ V₁) (%)')
        axs[1].set_title('b) Líkur á að ná lokamarkmiði')
        axs[1].legend(fontsize=8)
        fig.suptitle('Mynd 5. Næmni fyrir M (punktalína = grunnval M = 2M*)',
                     fontsize=10.5, fontweight='bold', x=0.02, ha='left')
        fig.tight_layout()
        _birta_mynd(fig, vista_i)
    return tafla, pivot


def profun_2024(il, I, grunnur=None, margfeldi=(2, 10, 100), V1=1600):
    """Nákvæm endurspilun ársins 2024. Skilar (prof, tafla, exante).

    2024 er prófunarár utan þjálfunar. Ex ante er miðað við desemberflokk 2023.
    """
    if 2024 not in I.index:
        raise ValueError('Ár 2024 vantar í mánaðarlegt innrennsli I.')
    Iar = I.loc[2024, list(range(1, 13))].to_numpy(dtype=float)
    c2024 = np.array([flokkur(Iar[m], m + 1, il.mork) for m in range(12)])
    des2023 = int(il.flokkar.loc[2023, 12])
    pi_des = np.eye(3)[des2023]
    prof, sam, exante = {}, {}, {}
    for k in margfeldi:
        if k == 2 and grunnur is not None and V1 in grunnur and np.isclose(grunnur[V1]['st'].M, 2*M_JAFN):
            sd = grunnur[V1]['sdp']
        else:
            sd = SDP(il, Stillingar(V1=V1, M=float(k)*M_JAFN)).leysa()
        df = keyra_ar(sd.stefnufall(), Iar, c2024, V1)
        prof[k] = df
        sam[f'M = {k}M*'] = {lykill: round(float(val), 1)
                             for lykill, val in samantekt_ars(df, V1).items()}
        ea = meta_slodir(sd.stefnufall(), il, V1, pi_des=pi_des)
        exante[k] = {
            'areidanleiki': float(ea['areidanleiki']),
            'E_afhent_GWh': float(ea['E_afhent_GWh']),
            'E_skert_GWh': float(ea['E_skert_GWh']),
            'P_V13_ge_V1': float(ea['P_V13_ge_V1']),
        }
    return prof, pd.DataFrame(sam), pd.DataFrame(exante).T.rename_axis('M/M*')


def _maeliraðir_2024(klst):
    """Rúmmál og framleiðsla 2024 úr sömu klukkustundagögnum og SDP-skýrslan."""
    rum = klst['rummal'].set_index('timi')['gildi']
    afl = klst['afl'].set_index('timi')['gildi']
    rum24 = rum[rum.index.year == 2024].resample('D').mean()
    afl24 = afl[afl.index.year == 2024].resample('MS').mean()
    return rum24, afl24


def teikna_profun_2024(prof, klst, V1=1600, vista_i=None):
    """Mynd 6: Líkanið á móti mældu vatnsrúmmáli og framleiðslu 2024."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    lit, _ = _litasnid()
    rum24, afl24 = _maeliraðir_2024(klst)
    manlok = pd.to_datetime([f'2024-{m:02d}-01' for m in range(1, 13)]) + pd.offsets.MonthEnd(0)
    fig, axs = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True,
                            gridspec_kw={'height_ratios': [2.2, 1]})
    ax = axs[0]
    ax.plot(rum24.index, rum24.to_numpy(), color=lit['blek2'], lw=1.4,
            label='Mælt rúmmál Hálslóns (dagsmeðaltal)')
    for k, color, ls in ((2, lit['blar'], '-'), (100, lit['app'], '--')):
        if k not in prof:
            continue
        ax.plot([pd.Timestamp('2024-01-01')] + list(manlok),
                [V1] + list(prof[k]['V_lok']), color=color, ls=ls, marker='o', ms=4,
                label=f'Líkan, SDP M = {k}M*' + (' (grunnval)' if k == 2 else ''))
    ax.axhline(V_MIN, color=lit['hraedd'], lw=0.8, ls='--')
    for a, b in (('2024-02-15', '2024-05-08'), ('2024-11-23', '2024-12-31')):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=lit['gulur'], alpha=0.15, lw=0)
    ax.text(pd.Timestamp('2024-02-18'), 2050, 'Raunveruleg skerðing\nN- og A-lands feb–maí', fontsize=7.5, color=lit['blek2'])
    ax.text(pd.Timestamp('2024-10-12'), 1300, 'Skerðing frá\n23. nóv.', fontsize=7.5, color=lit['blek2'])
    ax.set_ylim(0, 2450)
    ax.set_ylabel('Rúmmál (Mm³)')
    ax.legend(fontsize=7.5, loc='lower right')
    ax.set_title('Mynd 6. Prófunarárið 2024: líkan (537 MW) á móti raunverulegu lóni')
    ax = axs[1]
    ax.step(afl24.index, afl24.to_numpy(), where='post', color=lit['blek2'], lw=1.6,
            label='Raunveruleg framleiðsla')
    if 2 not in prof:
        raise ValueError('Mynd 6 þarf prófun fyrir M=2M*.')
    ax.step(afl24.index, FAST + prof[2]['x'].to_numpy(), where='post',
            color=lit['blar'], label='Líkan: 483 + x')
    ax.set_ylabel('Afl (MW)')
    ax.set_ylim(450, 660)
    ax.legend(fontsize=7.5, loc='lower right', ncol=2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    fig.tight_layout()
    return _birta_mynd(fig, vista_i)


def klukkustundaprof_2024(prof, klst, V1=1600, teikna=True, vista_i=None):
    """Mynd 7 og tafla: mánaðarleg 2024-afhending endurspiluð á klukkustundum."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    if 2 not in prof:
        raise ValueError('Klukkustundaprófið þarf prof[2] (grunnval M=2M*).')
    H = klst['H'].set_index('timi')['gildi']
    J = klst['J'].set_index('timi')['gildi']
    k24 = pd.DataFrame({'H': H, 'J': J})
    k24 = k24[k24.index.year == 2024].copy()
    for col in ('H', 'J'):
        k24[col] = k24[col].fillna(k24.groupby(k24.index.month)[col].transform('mean'))
    if k24[['H', 'J']].isna().any().any():
        raise ValueError('Innrennslisgögn vantar eftir mánaðarlega uppfyllingu.')
    inn = ((k24['H'] + k24['J']) * 0.0036).to_numpy()
    q = ((prof[2]['F'] + prof[2]['x']).to_numpy() / K)[k24.index.month - 1]
    Vk = np.empty(len(k24))
    v = float(V1)
    for i in range(len(k24)):
        v = min(V_MAX, v + inn[i] - 0.0036 * q[i])
        Vk[i] = v
    k24['V'] = Vk
    gr = k24['V'].groupby(k24.index.month)
    ix = np.arange(1, 13)
    tafla = pd.DataFrame({
        'klst í mán': k24.groupby(k24.index.month).size().reindex(ix).to_numpy(),
        'líkan, lok mán': prof[2]['V_lok'].to_numpy(),
        'klst, lok mán': gr.last().reindex(ix).to_numpy(),
        'klst, lægst í mán': gr.min().reindex(ix).to_numpy()}, index=ix).round(0)
    tafla['dýfa undir lok mán'] = (
        np.minimum(np.r_[V1, tafla['klst, lok mán'].to_numpy()[:-1]],
                   tafla['klst, lok mán'].to_numpy()) - tafla['klst, lægst í mán'].to_numpy()
    ).clip(min=0).round(0)
    tafla.index = MAN
    if teikna:
        lit, _ = _litasnid()
        manlok = pd.to_datetime([f'2024-{m:02d}-01' for m in range(1, 13)]) + pd.offsets.MonthEnd(0)
        fig, ax = plt.subplots(figsize=(8.0, 3.3))
        ax.plot(k24.index, Vk, color=lit['blar'], lw=1.2,
                label='Klukkustundaendurspilun (sama afl og í líkani)')
        ax.plot([pd.Timestamp('2024-01-01')] + list(manlok),
                [V1] + list(prof[2]['V_lok']), color=lit['app'], lw=0,
                marker='o', ms=5, label='Mánaðarlíkan (lok líkanmánaðar)')
        ax.axhline(V_MIN, color=lit['hraedd'], lw=0.8, ls='--')
        ax.set_ylim(0, 2400)
        ax.set_ylabel('Rúmmál (Mm³)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.set_title('Mynd 7. Sama stefna 2024 endurspiluð á klukkustundargögnum')
        ax.legend(fontsize=7.5, loc='center right')
        _birta_mynd(fig, vista_i)
    return tafla, k24


def ovissa_likindafylkja(grunnur, il, I_thj, n_urtok=100, frae=2026,
                         teikna=True, vista_i=None):
    """Mynd 8: Dirichlet-óvissa, fryst SDP-stefna vs endurleyst SDP."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(frae)
    alpha = 0.5
    ovissa, kassar = {}, {}
    for V1 in (1600, 1950):
        g = grunnur[V1]
        st, base = g['st'], g['sdp']
        fr, en, eftirsja = [], [], []
        for _ in range(n_urtok):
            Ps = np.array([[rng.dirichlet(il.talningar[t, i] + alpha)
                            for i in range(3)] for t in range(12)])
            ils = Innrennslislikan(il.mork, il.flokkar, il.talningar, Ps,
                                   il.Ibar, il.pi_des)
            a = meta_net(base.stefna, ils, V1, base.Vnet)
            sd = SDP(ils, st).leysa()
            e = meta_net(sd.stefna, ils, V1, sd.Vnet)
            fr.append(a)
            en.append(e)
            obj = lambda z: z['E_afhent_GWh'] / GWH_A_MWMAN - st.M * z['E_halli'] - st.mu * z['E_skortur_GWh'] / GWH_A_MWMAN
            eftirsja.append(obj(e) - obj(a))
        bil = lambda L, key: [round(float(np.percentile([z[key] for z in L], q)), 3)
                               for q in (5, 50, 95)]
        ovissa[V1] = {
            'fryst': {key: bil(fr, key) for key in ('E_afhent_GWh', 'P_V13_ge_V1', 'E_halli', 'areidanleiki')},
            'endurleyst': {key: bil(en, key) for key in ('E_afhent_GWh', 'P_V13_ge_V1', 'E_halli', 'areidanleiki')},
            'eftirsja_MWman_5_50_95': [round(float(np.percentile(eftirsja, q)), 2) for q in (5, 50, 95)]}
        kassar[V1] = (fr, en)
        ilu = Innrennslislikan(il.mork, il.flokkar, il.talningar,
                               np.full((12, 3, 3), 1/3), il.Ibar, il.pi_des)
        ru = meta_slodir(base.stefnufall(), ilu, V1)
        ovissa[V1]['oahadir_manudir_fryst'] = {
            'E_afhent_GWh': float(ru['E_afhent_GWh']),
            'areidanleiki': float(ru['areidanleiki']),
            'P_V13_ge_V1': float(ru['P_V13_ge_V1'])}
        # Ath: ársinnrennsli þarf g['Ip']; það er búið til úr ferlum ef vantar.
        rs = g['rs']
        Ip = g.get('Ip')
        if Ip is None:
            Ip = np.array([il.Ibar[t, rs['saga']['c'][:, t]] for t in range(12)]).T
        Iann, p = Ip.sum(1), rs['p']
        med = p @ Iann
        ovissa[V1]['arsinnrennsli_kedja'] = dict(
            medal=round(float(med), 0),
            sd=round(float(np.sqrt(p @ (Iann-med)**2)), 0),
            min=round(float(Iann.min()), 0), max=round(float(Iann.max()), 0),
            P_lt_3534=round(float(p @ (Iann < 3534)), 4))
    ovissa['saga_sd'] = round(float(I_thj.sum(axis=1).std(ddof=1)), 0)
    if teikna:
        lit, _ = _litasnid()
        fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.2))
        kw = dict(widths=0.7, patch_artist=True,
                  boxprops=dict(facecolor='#9ec5f4', color=lit['blek2']),
                  medianprops=dict(color=lit['app'], lw=2),
                  whiskerprops=dict(color=lit['blek2']),
                  capprops=dict(color=lit['blek2']),
                  flierprops=dict(ms=3, markeredgecolor=lit['daufur']))
        for k, V1 in enumerate((1600, 1950)):
            fr, en = kassar[V1]
            axs[0].boxplot([[z['E_afhent_GWh'] for z in fr],
                            [z['E_afhent_GWh'] for z in en]],
                           positions=[3*k, 3*k+1], **kw)
            axs[1].boxplot([[100*z['P_V13_ge_V1'] for z in fr],
                            [100*z['P_V13_ge_V1'] for z in en]],
                           positions=[3*k, 3*k+1], **kw)
        for ax in axs:
            ax.set_xticks([0, 1, 3, 4], ['fryst\n1600', 'endurleyst\n1600',
                                        'fryst\n1950', 'endurleyst\n1950'], fontsize=7.5)
        axs[0].set_ylabel('Vænt afhent orka (GWh/ár)')
        axs[0].set_title('a) Afhending')
        axs[1].set_ylabel('Pr(V₁₃ ≥ V₁) (%)')
        axs[1].set_title('b) Lokamarkmið')
        fig.suptitle(f'Mynd 8. Óvissa í líkindafylkjum: {n_urtok} Dirichlet-úrtök (α = 0,5)',
                     fontsize=10.5, fontweight='bold', x=0.02, ha='left')
        fig.tight_layout()
        _birta_mynd(fig, vista_i)
    return ovissa, kassar


def alagsprof_areidanleika(il, V1_gildi=range(900, 1426, 25),
                           teikna=True, vista_i=None):
    """Mynd 9: áreiðanleiki við lágar upphafsvatnsstöður."""
    import matplotlib.pyplot as plt
    linur = []
    for V1 in V1_gildi:
        ra = meta_slodir(stefna_alltaf, il, V1)
        rn = meta_slodir(stefna_aldrei, il, V1)
        lam, sd, rr, stada = finna_lam(il, Stillingar(V1=V1))
        ok = stada != 'engin_stefna'
        linur.append(dict(
            V1=V1, rel_alltaf=round(float(ra['areidanleiki']), 4),
            rel_aldrei=round(float(rn['areidanleiki']), 4), stada=stada,
            lam=round(float(lam), 3) if ok else None,
            rel_SDP=round(float(rr['areidanleiki']), 4) if ok else None,
            afhent_SDP=round(float(rr['E_afhent_GWh']), 1) if ok else None,
            skortur_SDP_GWh=round(float(rr['E_skortur_GWh']), 2) if ok else None,
            afhent_alltaf=round(float(ra['E_afhent_GWh']), 1)))
    tafla = pd.DataFrame(linur)
    if teikna:
        lit, _ = _litasnid()
        fig, ax = plt.subplots(figsize=(7.4, 3.3))
        ax.plot(tafla.V1, 100*tafla.rel_alltaf, color=lit['gr'], marker='o', ms=4,
                label='Alltaf afhenda')
        ax.plot(tafla.V1, 100*tafla.rel_aldrei, color=lit['app'], marker='o', ms=4,
                label='Aldrei afhenda (mesti mögulegi áreiðanleiki)')
        ok = tafla.rel_SDP.notna()
        ax.plot(tafla.V1[ok], 100*tafla.rel_SDP[ok], color=lit['blar'], marker='o', ms=4,
                label='SDP með leit að λ')
        ax.axhline(95, color=lit['blek2'], lw=0.8, ls='--')
        ax.text(min(V1_gildi)+5, 96.5, 'Krafa 95%', fontsize=7.5, color=lit['blek2'])
        ax.set_xlabel('Upphafsstaða V₁ (Mm³)')
        ax.set_ylabel('Pr(ár án skorts) (%)')
        ax.set_title('Mynd 9. Áreiðanleiki forgangsorku við lægri upphafsstöðu')
        ax.legend(fontsize=8, loc='lower right')
        _birta_mynd(fig, vista_i)
    return tafla


def yfirfallsgreining(grunnur, t_man, il=None, teikna=True, vista_i=None):
    """Mynd 10, sögulegar yfirfallsmælingar og möguleg aukasala úr yfirfalli."""
    import matplotlib.pyplot as plt
    yf = t_man.loc[t_man['yfirfall_Mm3'] > 1,
                    ['yfirfall_Mm3', 'afl_medal', 'klst_i_man']].copy()
    yf['orka_yfirfalls_GWh'] = yf['yfirfall_Mm3'] / A_VATN * GWH_A_MWMAN
    yf['laust_afl_690_MW'] = 690 - yf['afl_medal']
    yf['laust_rennsli_m3s'] = (690 - yf['afl_medal']) / K
    yf['yfirfall_m3s'] = yf['yfirfall_Mm3'] / (0.0036 * yf['klst_i_man'])
    umfram = {}
    for V1, g in grunnur.items():
        rs, p = g['rs'], g['rs']['p']
        xs, Vs = rs['saga']['x'], rs['saga']['V']
        Vb = np.concatenate([np.full((len(p), 1), float(V1)), Vs[:, :11]], axis=1)
        Ip = g.get('Ip')
        if Ip is None:
            if il is None:
                raise ValueError('Yfirfallsgreining þarf il ef Ip er ekki í grunnur.')
            Ip = np.array([il.Ibar[t, rs['saga']['c'][:, t]] for t in range(12)]).T
        S = np.maximum(0, Vb + Ip - A_VATN * (FAST + xs) - V_MAX)
        vol = p @ S.sum(1)
        umfram[V1] = {
            'E_yfirfall_Mm3': round(float(vol), 0),
            'orka_yfirfalls_GWh': round(float(vol/A_VATN*GWH_A_MWMAN), 0),
            **{f'aukasala_{cap}MW_GWh': round(float(
                p @ np.minimum(S/A_VATN, cap-(FAST+xs)).sum(1)*GWH_A_MWMAN), 0)
               for cap in (575, 690)}}
    umfram_tafla = pd.DataFrame(umfram)
    if teikna:
        lit, _ = _litasnid()
        ys = t_man['yfirfall_Mm3'].groupby(level='ar').sum()
        fig, ax = plt.subplots(figsize=(7.4, 3.0))
        ax.bar(ys.index, ys.to_numpy(), color=lit['blar'], width=0.62)
        for a, v in ys.items():
            ax.text(a, v + 25, f'{v:.0f}', ha='center', fontsize=7.5, color=lit['blek2'])
        ax.set_xticks(ys.index)
        ax.set_ylabel('Mælt yfirfall (Mm³/ár)')
        ax.set_title('Mynd 10. Mælt yfirfall Hálslóns 2015–2025')
        _birta_mynd(fig, vista_i)
    return yf, umfram_tafla


def sannprofa_grunnlausn(grunnur, il):
    """Grunnpróf: vatnsjafnvægi, líkindasummur og samræmi Bellman/ferla/nets.

    Ekki framkvæma dýra trjáleit eða netþéttleikapróf hér.
    """
    rows = []
    for V1, g in grunnur.items():
        rs, st, sd = g['rs'], g['st'], g['sdp']
        Ip = g.get('Ip')
        if Ip is None:
            Ip = np.array([il.Ibar[t, rs['saga']['c'][:, t]] for t in range(12)]).T
        jafnv = V1 + Ip.sum(1) - A_VATN*(12*FAST - rs['sum_D'] + rs['sum_x']) - rs['sum_S'] - rs['V13']
        rn = meta_net(sd.stefna, il, V1, sd.Vnet)
        rows.append(dict(V1=V1, Bellman_U1=float(sd.U1(V1)),
                         markgildi_ferlar=float(rs['p'] @ (
                             rs['sum_x']-st.mu*rs['sum_D']-st.M*rs['halli']-st.lam*rs['f'])),
                         jafnvaegi_max_frav=float(np.abs(jafnv).max()),
                         summa_likinda=float(rs['p'].sum()),
                         afhent_net_GWh=float(rn['E_afhent_GWh']),
                         afhent_ferlar_GWh=float(rs['E_afhent_GWh']),
                         halli_net=float(rn['E_halli']),
                         halli_ferlar=float(rs['E_halli']),
                         laegsta_V=float(rs['Vmin'].min())))
    return pd.DataFrame(rows).set_index('V1')
