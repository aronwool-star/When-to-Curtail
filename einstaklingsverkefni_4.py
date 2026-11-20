"""
Einstaklingsverkefni 4: Gildisítrun borin saman við stefnuítrun.

Höfundur: Sófus Máni Bender

Hópurinn leysir SDP-líkanið með bakvirkri lausn Bellman-jöfnunnar, sem er gildisítrun (value iteration, VI)
á endanlegum tímasjóndeildarhring. Hér er sama vandamál leyst með stefnuítrun (policy iteration, PI):

  1. upphafsstefna pi(t, c, f, V) á sama neti og hópurinn notar,
  2. mat á stefnu: J^pi reiknað aftur á bak með fastri stefnu (ein aðgerð í hverju ástandi, ekkert hámark),
  3. bæting: eitt Bellman-skref í öllum ástöndum og öllum mánuðum samtímis, miðað við J^pi,
  4. endurtekið þar til stefnan breytist ekki.

Líkani hópsins (curtail_sdp.py) er ekki breytt. Öllum forsendum hópsins er haldið: ástand (V, c, f),
net dV = 1 Mm3 og dx = 0,5 MW, M = 2M*, mu = 10, jafnteflisregla hópsins og lambda sem ytri leit
hópsins skilaði.
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import curtail_sdp as cs

TOL_BREYT = 1e-9          # MW: stefna telst óbreytt ef engin aðgerð breytist meira en þetta
UPPHAF = ["alltaf", "aldrei"]
REGLUR = ["hopur", "halda"]

DALKAR = {
    "V1": "V₁", "lam": "λ", "adferd": "Aðferð", "upphaf": "Upphafsstefna", "regla": "Jafnteflisregla",
    "mat": "Stefnumöt", "timi_s": "Tími (s)", "U1": "U₁ (MW-mán)", "max_J_frav": "max |J − J_VI|",
    "ostodur_frav": "Ástönd með aðra stefnu", "max_Q_bil": "Mesti Q-munur", "afhent_GWh": "Afhent (GWh)",
    "areidanleiki": "Áreiðanleiki", "P_V13": "Pr(V₁₃ ≥ V₁)", "E_halli": "E[halli] (Mm³)",
    "afhent_net": "Afhent, net (GWh)", "areid_net": "Áreiðanleiki, net",
}


# ----------------------------------------------------------------------------
# 1. Stefnuítrun
# ----------------------------------------------------------------------------
def _framhald(G, t, c, Vn, fn, Vnet):
    """G_t(V', c, f') með línulegri brúun; Vn getur verið (n,) eða (n, a), fn er (n,)."""
    out = np.empty_like(Vn)
    for fi in (0, 1):
        r = fn == fi
        if r.any():
            out[r] = np.interp(Vn[r], Vnet, G[t, c, fi])
    return out


def q_gildi(il, st, Vnet, t, c, f, X, G):
    """Q-gildi aðgerða X, (n,) eða (n, a), í ástöndunum (V = Vnet, c, f) í mánuði t miðað við framhaldsgildi G.
    Aðgerðir eru klipptar að xmax, nákvæmlega eins og í líkani hópsins."""
    W, F, D, xmax = cs.manudur(Vnet, il.Ibar[t, c])
    fn = np.maximum(f, (D > 1e-9).astype(int))      # skortur veltur ekki á x: forgangsorka fær vatnið fyrst
    tvivitt = X.ndim == 2
    X = np.minimum(X, xmax[:, None] if tvivitt else xmax)
    Vn, _ = cs.lok_manadar(W[:, None] if tvivitt else W, F[:, None] if tvivitt else F, X)
    Dm = D[:, None] if tvivitt else D
    return X, X - st.mu * Dm + _framhald(G, t, c, Vn, fn, Vnet), xmax


def lokagildi(st, Vnet):
    """J_13(V, f) = -M max(0, V1 - V) - lambda f, eins og hjá hópnum."""
    G11 = np.zeros((3, 2, len(Vnet)))
    for fi in (0, 1):
        G11[:, fi] = -st.M * np.maximum(0.0, st.V1 - Vnet) - st.lam * fi
    return G11


def meta_stefnu(il, st, Vnet, pol):
    """Mat á stefnu: J^pi og G^pi aftur á bak með fastri stefnu pol[t, c, f, iV] (ekkert hámark)."""
    J = np.zeros((12, 3, 2, len(Vnet)))
    G = np.zeros_like(J)
    G[11] = lokagildi(st, Vnet)
    for t in range(11, -1, -1):
        for c in range(3):
            for f in (0, 1):
                _, J[t, c, f], _ = q_gildi(il, st, Vnet, t, c, f, pol[t, c, f], G)
        if t > 0:
            G[t - 1] = np.einsum("ij,jfv->ifv", il.P[t], J[t])
    return J, G


def _allar_adgerdir(il, Vnet, Xnet, t, c):
    n = len(Vnet)
    return np.concatenate([np.broadcast_to(Xnet, (n, len(Xnet))),
                           cs.manudur(Vnet, il.Ibar[t, c])[3][:, None]], axis=1)


def baeta(il, st, Vnet, Xnet, pol, G, regla="hopur"):
    """Bæting: eitt Bellman-skref í öllum ástöndum miðað við G = G^pi.
    regla 'hopur': regla hópsins, stærsta x af þeim sem eru innan st.jafntefli frá hámarki.
    regla 'halda': hefðbundin PI-regla, núverandi aðgerð haldið nema önnur sé betri um meira en
                   st.jafntefli; annars fyrsta (minnsta) x sem nær hámarki."""
    ny = np.empty_like(pol)
    n = len(Vnet)
    for t in range(12):
        for c in range(3):
            Xall = _allar_adgerdir(il, Vnet, Xnet, t, c)
            for f in (0, 1):
                X, Q, _ = q_gildi(il, st, Vnet, t, c, f, Xall, G)
                Qmax = Q.max(axis=1)
                if regla == "hopur":
                    ny[t, c, f] = np.where(Q >= Qmax[:, None] - st.jafntefli, X, -1.0).max(axis=1)
                else:
                    xcur, Qcur, _ = q_gildi(il, st, Vnet, t, c, f, pol[t, c, f], G)
                    besta = X[np.arange(n), Q.argmax(axis=1)]
                    ny[t, c, f] = np.where(Qcur >= Qmax - st.jafntefli, xcur, besta)
    return ny


def U1_ur_J(il, Vnet, J, V1):
    """Vænt gildi í byrjun árs (áður en janúarinnrennsli sést), sama formúla og SDP.U1 hjá hópnum."""
    return float(il.pi_des @ il.P[0] @ np.array([np.interp(V1, Vnet, J[0, c, 0]) for c in range(3)]))


def upphafsstefna(il, Vnet, nafn):
    """'alltaf': 54 MW þegar vatn leyfir (x = xmax). 'aldrei': x = 0 alls staðar."""
    pol = np.zeros((12, 3, 2, len(Vnet)))
    if nafn == "alltaf":
        for t in range(12):
            for c in range(3):
                pol[t, c, :] = cs.manudur(Vnet, il.Ibar[t, c])[3]
    return pol


def stefnuitrun(il, st, upphaf, regla="hopur", vi_stefna=None, max_itr=60):
    """Stefnuítrun. upphaf er 'alltaf', 'aldrei' eða stefnutafla. Skilar dict með lokastefnu, J, G og ferli."""
    sd = cs.SDP(il, st)                         # aðeins notað til að fá sömu net og hópurinn
    Vnet, Xnet = sd.Vnet, sd.Xnet
    pol = upphafsstefna(il, Vnet, upphaf) if isinstance(upphaf, str) else upphaf.copy()
    saga, t_alls = [], time.perf_counter()
    for k in range(1, max_itr + 1):
        ta = time.perf_counter()
        J, G = meta_stefnu(il, st, Vnet, pol)
        tb = time.perf_counter()
        ny = baeta(il, st, Vnet, Xnet, pol, G, regla)
        tc = time.perf_counter()
        breytt = np.abs(ny - pol) > TOL_BREYT
        lina = dict(itr=k, U1=U1_ur_J(il, Vnet, J, st.V1), breytt=int(breytt.sum()),
                    breytt_man=breytt.sum(axis=(1, 2, 3)), t_mat=tb - ta, t_baet=tc - tb)
        if vi_stefna is not None:
            lina["frav_VI_man"] = (np.abs(pol - vi_stefna) > 1e-6).mean(axis=(1, 2, 3))
        saga.append(lina)
        if not breytt.any():
            break
        pol = ny
    return dict(stefna=pol, J=J, G=G, saga=saga, itr=len(saga), timi=time.perf_counter() - t_alls, Vnet=Vnet)


def sem_sdp(il, st, nidurst):
    """Pakkar PI-lausn í SDP-hlut hópsins svo matsföll hópsins (meta_slodir o.fl.) virki óbreytt."""
    s = cs.SDP(il, st)
    s.J, s.G, s.stefna = nidurst["J"], nidurst["G"], nidurst["stefna"]
    return s


# ----------------------------------------------------------------------------
# 2. Tilraunir
# ----------------------------------------------------------------------------
def sannprofa_mat(grunnur, il):
    """Mat á VI-stefnu hópsins á að gefa nákvæmlega J frá VI, og PI sem byrjar þar á að hætta strax."""
    lin = []
    for V1, g in grunnur.items():
        st, sdp_vi = g["st"], g["sdp"]
        J_mat, _ = meta_stefnu(il, st, sdp_vi.Vnet, sdp_vi.stefna)
        byrja = stefnuitrun(il, st, sdp_vi.stefna)
        lin.append({"V₁": V1, "λ": st.lam, "max |J(mat á VI-stefnu) − J_VI|": float(np.abs(J_mat - sdp_vi.J).max()),
                    "Stefnumöt PI frá VI-stefnu": byrja["itr"]})
    return pd.DataFrame(lin)


def _meta_net(stefna, il, V1, Vnet):
    """Stefnutaflan sjálf metin með líkindadreifingu á netinu, ef fallið er til í líkani hópsins."""
    if hasattr(cs, "meta_net"):
        r = cs.meta_net(stefna, il, V1, Vnet)
        return r["E_afhent_GWh"], r["areidanleiki"]
    return np.nan, np.nan


def _bera_saman_eitt(V1, st, sdp_vi, il, geymsla, endurtekningar=5):
    """VI á móti PI (2 upphafsstefnur x 2 jafnteflisreglur) fyrir eina upphafsstöðu."""
    tvi = []
    for _ in range(endurtekningar):
        t0 = time.perf_counter()
        cs.SDP(il, st).leysa()
        tvi.append(time.perf_counter() - t0)
    rs = cs.meta_slodir(sdp_vi.stefnufall(), il, V1)
    an, ar = _meta_net(sdp_vi.stefna, il, V1, sdp_vi.Vnet)
    lin = [dict(V1=V1, lam=st.lam, adferd="VI", upphaf="–", regla="hopur", mat=1, timi_s=np.median(tvi),
                U1=sdp_vi.U1(V1), max_J_frav=0.0, ostodur_frav=0, max_Q_bil=0.0,
                afhent_GWh=rs["E_afhent_GWh"], areidanleiki=rs["areidanleiki"], P_V13=rs["P_V13_ge_V1"],
                E_halli=rs["E_halli"], afhent_net=an, areid_net=ar)]
    for upphaf in UPPHAF:
        for regla in REGLUR:
            res = stefnuitrun(il, st, upphaf, regla, vi_stefna=sdp_vi.stefna)
            geymsla[(upphaf, regla)] = res
            s_pi = sem_sdp(il, st, res)
            rs = cs.meta_slodir(s_pi.stefnufall(), il, V1)
            an, ar = _meta_net(res["stefna"], il, V1, res["Vnet"])
            # Er stefnumunurinn jafntefli? Q-gildi beggja aðgerða reiknuð með G frá VI.
            frav = np.abs(res["stefna"] - sdp_vi.stefna) > 1e-6
            Qbil = 0.0
            for t in range(12):
                for c in range(3):
                    for f in (0, 1):
                        r_ = frav[t, c, f]
                        if r_.any():
                            _, q_vi, _ = q_gildi(il, st, sdp_vi.Vnet, t, c, f, sdp_vi.stefna[t, c, f], sdp_vi.G)
                            _, q_pi, _ = q_gildi(il, st, sdp_vi.Vnet, t, c, f, res["stefna"][t, c, f], sdp_vi.G)
                            Qbil = max(Qbil, float(np.abs(q_vi - q_pi)[r_].max()))
            lin.append(dict(V1=V1, lam=st.lam, adferd="PI", upphaf=upphaf, regla=regla, mat=res["itr"],
                            timi_s=res["timi"], U1=s_pi.U1(V1), max_J_frav=float(np.abs(res["J"] - sdp_vi.J).max()),
                            ostodur_frav=int(frav.sum()), max_Q_bil=Qbil, afhent_GWh=rs["E_afhent_GWh"],
                            areidanleiki=rs["areidanleiki"], P_V13=rs["P_V13_ge_V1"], E_halli=rs["E_halli"],
                            afhent_net=an, areid_net=ar))
    return lin


def _snida(df):
    nakvaemni = {"lam": 4, "timi_s": 2, "U1": 3, "max_J_frav": 6, "max_Q_bil": 6, "afhent_GWh": 2,
                 "areidanleiki": 4, "P_V13": 3, "E_halli": 2, "afhent_net": 2, "areid_net": 4}
    return df.round(nakvaemni).rename(columns=DALKAR)


def bera_saman_VI_PI(grunnur, il):
    """Samanburður fyrir allar upphafsstöður hópsins. Skilar (tafla, PI) þar sem PI[V1][(upphaf, regla)]."""
    PI, lin = {}, []
    for V1, g in grunnur.items():
        PI[V1] = {}
        lin += _bera_saman_eitt(V1, g["st"], g["sdp"], il, PI[V1])
    return _snida(pd.DataFrame(lin)), PI


def alagstilvik(il, V1=1100):
    """Upphafsstaða þar sem 95% krafan bindur: lambda fundið með leit hópsins og síðan fest."""
    lam, sdp_a, _, stada = cs.finna_lam(il, cs.Stillingar(V1=V1), krafa=0.95)
    print(f"V₁ = {V1}: λ* = {lam:.4f} ({stada})")
    geymsla = {}
    return _snida(pd.DataFrame(_bera_saman_eitt(V1, sdp_a.st, sdp_a, il, geymsla))), geymsla


def ferill_itrana(PI, V1=1600, regla="hopur"):
    """Fjöldi ástanda sem breytast í bætingarskrefi hverrar ítrunar, eftir mánuðum, og gildi stefnunnar."""
    toflur = []
    for upphaf in UPPHAF:
        sg = PI[V1][(upphaf, regla)]["saga"]
        d = pd.DataFrame([s["breytt_man"] for s in sg], columns=cs.MAN)
        d.insert(0, "U₁ stefnu", [round(s["U1"], 2) for s in sg])
        d.index = pd.MultiIndex.from_tuples([(upphaf, s["itr"]) for s in sg], names=["Upphafsstefna", "Ítrun"])
        toflur.append(d)
    return pd.concat(toflur)


def timaskipting(PI):
    """Meðaltími á ítrun: mat á stefnu á móti bætingarskrefi."""
    lin = []
    for V1, d in PI.items():
        for (upphaf, regla), res in d.items():
            lin.append({"V₁": V1, "Upphafsstefna": upphaf, "Jafnteflisregla": regla,
                        "Mat á stefnu (s/ítrun)": np.mean([s["t_mat"] for s in res["saga"]]),
                        "Bæting (s/ítrun)": np.mean([s["t_baet"] for s in res["saga"]])})
    d = pd.DataFrame(lin)
    return d.groupby("V₁")[["Mat á stefnu (s/ítrun)", "Bæting (s/ítrun)"]].mean().round(4)


def teikna_bylgju(PI, V1=1600):
    """Hlutfall ástanda hvers mánaðar þar sem stefna ítrunar k víkur frá VI-stefnu hópsins."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, upphaf in zip(axs, UPPHAF):
        sg = PI[V1][(upphaf, "hopur")]["saga"]
        M_ = np.array([s["frav_VI_man"] for s in sg]) * 100
        im = ax.imshow(M_, aspect="auto", cmap="Blues", vmin=0, vmax=100,
                       extent=[0.5, 12.5, len(M_) + 0.5, 0.5], interpolation="nearest")
        for k in range(M_.shape[0]):
            for t in range(12):
                if M_[k, t] > 0:
                    ax.text(t + 1, k + 1, f"{M_[k, t]:.0f}" if M_[k, t] >= 0.5 else "<1", ha="center",
                            va="center", fontsize=8, color="white" if M_[k, t] > 55 else "black")
        ax.set_xticks(range(1, 13), cs.MAN)
        ax.set_yticks(range(1, len(M_) + 1))
        ax.set_xlabel("Mánuður")
        ax.set_ylabel("Ítrun k (stefnan sem er metin)")
        ax.set_title(f"Upphafsstefna: „{upphaf}“")
    fig.colorbar(im, ax=axs, shrink=0.85, label="% ástanda þar sem stefna víkur frá VI")
    fig.suptitle(f"Stefnuítrun, V₁ = {V1} Mm³: rétt stefna dreifist aftur á bak frá desember", y=1.03)
    plt.show()
    return fig


def teikna_samleitni(PI, grunnur):
    """Munur á gildi VI-stefnu og stefnu ítrunar k, fyrir hverja upphafsstöðu."""
    fig, axs = plt.subplots(1, len(grunnur), figsize=(13, 4))
    for ax, (V1, g) in zip(np.atleast_1d(axs), grunnur.items()):
        u_vi = g["sdp"].U1(V1)
        for upphaf, mk in (("alltaf", "o"), ("aldrei", "s")):
            sg = PI[V1][(upphaf, "hopur")]["saga"]
            ax.plot([s["itr"] for s in sg], [u_vi - s["U1"] for s in sg], marker=mk, label=f"Upphaf „{upphaf}“")
        ax.set_yscale("symlog", linthresh=1e-2)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Ítrun k")
        ax.set_title(f"V₁ = {V1} Mm³")
        ax.grid(alpha=0.3)
    np.atleast_1d(axs)[0].set_ylabel("U₁(VI) − U₁(stefna k)  (MW-mán)")
    np.atleast_1d(axs)[0].legend()
    fig.suptitle("Gildi stefnu í hverri ítrun stefnuítrunar, borið saman við gildisítrun")
    plt.tight_layout()
    plt.show()
    return fig


def jafnteflagreining(grunnur, il):
    """Hvar er besta aðgerð ekki einkvæm? Fyrir hvert ástand VI-lausnarinnar eru fundin öll x sem eru innan
    st.jafntefli frá besta Q-gildi. Ástönd með xmax = 0 (ekkert val) eru talin sér."""
    jt = []
    for V1, g in grunnur.items():
        st, sdp_vi = g["st"], g["sdp"]
        Vnet, Xnet = sdp_vi.Vnet, sdp_vi.Xnet
        for t in range(12):
            for c in range(3):
                Xall = _allar_adgerdir(il, Vnet, Xnet, t, c)
                for f in (0, 1):
                    X, Q, xmax = q_gildi(il, st, Vnet, t, c, f, Xall, sdp_vi.G)
                    naer = np.where(Q >= Q.max(1)[:, None] - st.jafntefli, X, np.nan)
                    breidd = np.nanmax(naer, 1) - np.nanmin(naer, 1)
                    jt.append(dict(V1=V1, t=t, f=f, ekkert_val=int((xmax < 1e-9).sum()),
                                   jafntefli=int(((breidd > 1e-9) & (xmax >= 1e-9)).sum()), alls=len(Vnet)))
    jt = pd.DataFrame(jt)
    samant = jt[jt.f == 0].groupby("V1")[["ekkert_val", "jafntefli", "alls"]].sum()
    samant["% ekkert val"] = (100 * samant["ekkert_val"] / samant["alls"]).round(2)
    samant["% jafntefli"] = (100 * samant["jafntefli"] / samant["alls"]).round(2)
    samant = samant.rename(columns={"ekkert_val": "Ástönd án vals (xmax = 0)", "jafntefli": "Jafnteflisástönd",
                                    "alls": "Ástönd (f = 0)"}).rename_axis("V₁")
    eftir_man = jt[jt.f == 0].groupby(["V1", "t"])["jafntefli"].sum().unstack("t")
    eftir_man.columns = cs.MAN
    return samant, eftir_man.rename_axis("V₁")
