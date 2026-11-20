
"""
Hluti 2 – Flokkun innrennslis og mánaðarbundin líkindafylki.

Keyrt úr Colab eftir hluta 1:

%run -i "/content/drive/MyDrive/When_to_Curtail/hluti_2.py"

Síðan eru föllin keyrð í aðskildum reitum:

2.1: mork, flokkar = hluti_2_1()
2.2: talningar, P_emp = hluti_2_2(flokkar)
2.3: P_man, stadgenglar = hluti_2_3(talningar, P_emp, mork)
2.4: I_flokkar, upphafsflokkur_2024 = hluti_2_4(
         mork, flokkar, talningar, P_man
     )

Krefst þess að I_thj hafi verið skilgreint í hluta 1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# Sameiginlegar fastar
FLOKKAR = ["Lítið", "Miðlungs", "Mikið"]
N_FLOKKAR = 3


def _gogn():
    """Sækir og sannprófar gögn úr hluta 1."""

    if "I_thj" not in globals():
        raise RuntimeError(
            "Keyrðu fyrst hluta 1 svo I_thj sé til."
        )

    gogn = globals()["I_thj"]

    man = globals().get(
        "MAN",
        ["jan", "feb", "mar", "apr", "maí", "jún",
         "júl", "ágú", "sep", "okt", "nóv", "des"]
    )

    ar = globals().get(
        "THJALFUNARAR",
        list(range(2015, 2024))
    )

    assert list(gogn.columns) == list(range(1, 13)), (
        "Dálkar I_thj verða að vera 1–12."
    )

    assert list(gogn.index) == list(ar) == list(range(2015, 2024)), (
        "I_thj verður að innihalda 2015–2023, en ekki 2024."
    )

    assert gogn.notna().all().all(), (
        "Innrennsli vantar í þjálfunargögnin."
    )

    return gogn, man, ar


def finna_flokk(gildi, man, mork=None):
    """Skilar 0, 1 eða 2 fyrir innrennsli tiltekins mánaðar."""

    if mork is None:
        if "mork" not in globals():
            raise RuntimeError(
                "Keyrðu fyrst hluti_2_1() til að fá flokkamörk."
            )
        mork = globals()["mork"]

    nedri = mork.loc[1 / 3, man]
    efri = mork.loc[2 / 3, man]

    if gildi <= nedri:
        return 0
    elif gildi <= efri:
        return 1
    else:
        return 2


# ---------------------------------------------------------
# 2.1 – Flokkun innrennslis
# ---------------------------------------------------------

def hluti_2_1():
    """Reiknar flokkamörk og flokkar sögulegt innrennsli."""

    I_thj, MAN, _ = _gogn()

    # Reiknum flokkamörk hvers mánaðar
    mork = I_thj.quantile([1 / 3, 2 / 3], axis=0)

    # Flokkum sögulegt innrennsli
    flokkar = pd.DataFrame(
        {
            man: I_thj[man].apply(
                lambda x: finna_flokk(x, man, mork)
            )
            for man in range(1, 13)
        },
        index=I_thj.index
    )

    print("=== 2.1 Flokkun innrennslis 2015–2023 ===")

    print(
        flokkar.replace(
            dict(enumerate(FLOKKAR))
        ).to_string()
    )

    print("\nFlokkamörk (Mm³), neðri | efri:")

    for man in range(1, 13):
        print(
            f"{MAN[man-1]:>3}: "
            f"{mork.loc[1/3, man]:7.1f} | "
            f"{mork.loc[2/3, man]:7.1f}"
        )

    print("\nFjöldi ára í hverjum flokki:")

    for man in range(1, 13):
        fjoldi = flokkar[man].value_counts()

        print(
            f"{MAN[man-1]:>3}: "
            + " / ".join(
                f"{FLOKKAR[i]}={fjoldi.get(i, 0)}"
                for i in range(N_FLOKKAR)
            )
        )

    return mork, flokkar


# ---------------------------------------------------------
# 2.2 – Mat á mánaðarbundnum líkindafylkjum
# ---------------------------------------------------------

def hluti_2_2(flokkar):
    """Reiknar umskiptatalningar og sögulegar líkur."""

    _, _, THJALFUNARAR = _gogn()

    talningar = {}

    for man in range(1, 13):

        fylki = np.zeros(
            (N_FLOKKAR, N_FLOKKAR),
            dtype=int
        )

        if man == 1:
            # Desember fyrra árs -> janúar næsta árs
            for ar in THJALFUNARAR[1:]:
                fyrri = int(flokkar.loc[ar - 1, 12])
                nuverandi = int(flokkar.loc[ar, 1])

                fylki[fyrri, nuverandi] += 1

        else:
            # Fyrri mánuður -> núverandi mánuður
            for ar in THJALFUNARAR:
                fyrri = int(flokkar.loc[ar, man - 1])
                nuverandi = int(flokkar.loc[ar, man])

                fylki[fyrri, nuverandi] += 1

        talningar[man] = fylki

    # Reiknum líkur úr talningunum
    P_emp = {}

    for man, fylki in talningar.items():

        P = np.zeros(
            (N_FLOKKAR, N_FLOKKAR),
            dtype=float
        )

        for i in range(N_FLOKKAR):
            samtals = fylki[i].sum()

            if samtals:
                P[i] = fylki[i] / samtals

        P_emp[man] = P

    print("\n=== 2.2 Dæmi: júní -> júlí ===")

    print("Talningar:")
    print(
        pd.DataFrame(
            talningar[7],
            index=FLOKKAR,
            columns=FLOKKAR
        ).to_string()
    )

    print("\nLíkindafylki fyrir júlí:")

    print(
        pd.DataFrame(
            P_emp[7],
            index=FLOKKAR,
            columns=FLOKKAR
        ).round(3).to_string()
    )

    return talningar, P_emp


# ---------------------------------------------------------
# 2.3 – Meðhöndlun tómra raða
# ---------------------------------------------------------

def hluti_2_3(talningar, P_emp, mork):
    """Notar normaldreifingu þar sem söguleg umskipti vantar."""

    I_thj, MAN, _ = _gogn()

    P_man = {}
    stadgenglar = []

    for man in range(1, 13):

        P = P_emp[man].copy()

        # Metum normaldreifingu mánaðarins
        mu = I_thj[man].mean()
        sigma = I_thj[man].std(ddof=1)

        nedri = mork.loc[1 / 3, man]
        efri = mork.loc[2 / 3, man]

        if sigma > 0:

            p_litid = stats.norm.cdf(
                nedri, mu, sigma
            )

            p_mid = (
                stats.norm.cdf(efri, mu, sigma)
                - p_litid
            )

            p_mikid = (
                1 - stats.norm.cdf(efri, mu, sigma)
            )

            p_normal = np.array([
                p_litid,
                p_mid,
                p_mikid
            ])

        else:
            p_normal = np.array([0.0, 1.0, 0.0])

        # Fyllum aðeins í raðir án umskipta
        for i in range(N_FLOKKAR):

            if talningar[man][i].sum() == 0:
                P[i] = p_normal

                stadgenglar.append(
                    (MAN[man - 1], FLOKKAR[i])
                )

        P_man[man] = P

    print("\n=== 2.3 Meðhöndlun tómra raða ===")

    print("Líkindafylki tilbúin:", len(P_man))

    if stadgenglar:
        for man, flokkur in stadgenglar:
            print(
                f"Normaldreifing notuð: {man} – {flokkur}"
            )
    else:
        print("Engar tómar raðir fundust.")

    return P_man, stadgenglar


# ---------------------------------------------------------
# 2.4 – Sannprófun og myndræn framsetning
# ---------------------------------------------------------

def hluti_2_4(mork, flokkar, talningar, P_man):
    """Sannprófar líkindafylki og reiknar meðalinnrennsli flokka."""

    I_thj, MAN, _ = _gogn()

    # Sannprófum öll tólf líkindafylkin
    assert len(P_man) == 12
    assert 2024 not in flokkar.index

    for man in range(1, 13):

        P = P_man[man]

        assert P.shape == (3, 3)
        assert np.isfinite(P).all()
        assert ((P >= 0) & (P <= 1)).all()

        assert np.allclose(
            P.sum(axis=1), 1
        ), f"Raðsummur ekki 1: {MAN[man-1]}"

        assert talningar[man].sum() == (
            8 if man == 1 else 9
        )

    print("\n=== 2.4 Sannprófun ===")
    print("Öll 12 líkindafylkin standast sannprófun.")

    # Myndræn framsetning líkindafylkja
    fig, axs = plt.subplots(
        3, 4, figsize=(14, 10)
    )

    for man, ax in enumerate(axs.flat, start=1):

        P = P_man[man]

        ax.imshow(
            P,
            vmin=0,
            vmax=1,
            cmap="Blues"
        )

        ax.set_title(MAN[man - 1])

        ax.set_xticks(
            range(3), ["L", "M", "H"]
        )

        ax.set_yticks(
            range(3), ["L", "M", "H"]
        )

        for i in range(3):
            for j in range(3):

                gildi = P[i, j]

                ax.text(
                    j, i,
                    f"{gildi:.2f}",
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if gildi > 0.5
                        else "black"
                    )
                )

    fig.supxlabel(
        "Flokkur núverandi mánaðar (L, M, H)"
    )

    fig.supylabel(
        "Flokkur fyrri mánaðar (L, M, H)"
    )

    fig.suptitle(
        "Mánaðarbundin líkindafylki",
        fontsize=15
    )

    plt.tight_layout()
    plt.show()

    # Meðalinnrennsli innan hvers flokks
    I_flokkar = np.zeros(
        (12, N_FLOKKAR)
    )

    for man in range(1, 13):
        for i in range(N_FLOKKAR):

            I_flokkar[man - 1, i] = (
                I_thj.loc[
                    flokkar[man] == i, man
                ].mean()
            )

    assert np.isfinite(I_flokkar).all()

    print("\nMeðalinnrennsli hvers flokks (Mm³):")

    print(
        pd.DataFrame(
            I_flokkar,
            index=MAN,
            columns=FLOKKAR
        ).round(1).to_string()
    )

    # Desember 2023 er þekktur fyrir prófun 2024
    upphafsflokkur_2024 = finna_flokk(
        I_thj.loc[2023, 12],
        12,
        mork
    )

    print(
        "\nDesemberflokkur fyrir prófun 2024:",
        FLOKKAR[upphafsflokkur_2024]
    )

    return I_flokkar, upphafsflokkur_2024
