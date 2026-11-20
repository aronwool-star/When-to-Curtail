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
