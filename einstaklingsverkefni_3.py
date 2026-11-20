"""Einstaklingsverkefni 3 – Er stefna hópsins 95% áreiðanleg?

Notkun í Colab eftir hluta 1–5:
    import einstaklingsverkefni_3 as ev3
    tafla = ev3.ohad_mat(grunnur, il)
    loyo = ev3.leave_one_year_out(I_thj, I_sdp)
    naemni = ev3.breytt_thjalfun(I_thj, grunnur, il)

ÁRÍÐANDI: ohad_mat notar aðra reikniaðferð (framvirka dreifingu á
vatnsstöðuneti) en nákvæma ferlaupptalningu hópsins. Netmat er nálgun.
LOYO endurmetur öll líkindafylki og endurleysir SDP fyrir hvert útilokað ár.
Hvert útilokað ár er aldrei notað til mats á eigin líkansútfærslu.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import curtail_sdp as cs


def hermun_areidanleika(grunnur, il, n=3000, frae=2026):
    """Óháð Monte Carlo-hermun með FRYSTUM SDP-stefnum hópsins.

    Líkurnar eru metnar með slembiúrtaki af ársferlum úr Markov-fylkjum;
    ekki er notuð upptalning ársferla eða framvirk líkindaútbreiðsla.
    Wilson-öryggisbil er reiknað fyrir ár án forgangsskorts.
    """
    if n < 1:
        raise ValueError('n verður að vera jákvæð heiltala.')
    rng = np.random.default_rng(frae)
    rows = []
    for V1, g in sorted(grunnur.items()):
        sd = g['sdp']
        des = rng.choice(3, size=n, p=il.pi_des)
        V = np.full(n, float(V1))
        f = np.zeros(n, dtype=np.int8)
        skert = np.zeros(n)
        for t in range(12):
            c = np.empty(n, dtype=np.int8)
            for j in range(3):
                ix = (des == j)
                if ix.any():
                    c[ix] = rng.choice(3, size=ix.sum(), p=il.P[t, j])
            I = il.Ibar[t, c]
            x = sd.stefnufall()(t, V, I, c, f)
            W, F, D, xmax = cs.manudur(V, I)
            x = np.clip(x, 0.0, xmax)
            V, _ = cs.lok_manadar(W, F, x)
            f = np.maximum(f, (D > 1e-9).astype(np.int8))
            skert += cs.AUKA - x
            des = c
        success = int((f == 0).sum())
        p = success / n
        # Wilson 95% öryggisbil; forðast villandi 100% vissu við 0 bilanir.
        z = 1.959963984540054
        den = 1 + z * z / n
        mid = (p + z * z / (2 * n)) / den
        half = z / den * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        rows.append({
            'V1': V1, 'Hermuð ár': n, 'Ár án skorts': success,
            'Ár með skorti': n - success,
            'Áreiðanleiki hermunar (%)': 100 * p,
            '95% bil neðra (%)': 100 * (mid - half),
            '95% bil efra (%)': 100 * (mid + half),
            'Áreiðanleiki hóps (%)': 100 * g['rs']['areidanleiki'],
            'Vænt skerðing hermunar (GWh)': float(skert.mean() * cs.GWH_A_MWMAN),
        })
    return pd.DataFrame(rows).set_index('V1')


def ohad_mat(grunnur, il):
    """Framvirkt ástandanet vs upptalning ferla – fryst stefna hópsins."""
    rows = []
    for V1, g in sorted(grunnur.items()):
        sd = g['sdp']
        net = cs.meta_net(sd.stefna, il, V1, sd.Vnet)
        ferlar = g['rs']
        rows.append({
            'V1': V1,
            'Áreiðanleiki hóps (%)': 100 * ferlar['areidanleiki'],
            'Áreiðanleiki óháður (%)': 100 * net['areidanleiki'],
            'Munur (prósentustig)': 100 * (net['areidanleiki'] - ferlar['areidanleiki']),
            'Afhending hóps (GWh)': ferlar['E_afhent_GWh'],
            'Afhending óháð (GWh)': net['E_afhent_GWh'],
            '95% staðist – óháð': bool(net['areidanleiki'] >= 0.95),
        })
    return pd.DataFrame(rows).set_index('V1')


def _leidbeining_stillingar(dV, dx):
    if dV <= 0 or dx <= 0:
        raise ValueError('dV og dx verða að vera jákvæð.')
    return {'dV': float(dV), 'dx': float(dx)}


def leave_one_year_out(I_thj: pd.DataFrame, I_all: pd.DataFrame, V1=1600,
                       dV=5.0, dx=2.0, krafa=0.95, frae=2026,
                       framvinda=True):
    """Útilokar eitt ár 2015–2023; endurmetur og leysir, prófar á útilokaða árinu.

    dV=5, dx=2 er hraðari NÁLGUN, ekki nákvæmlega sama net og hópurinn.
    Endanlega má kalla með dV=1.0 og dx=0.5.
    Notar núverandi (raunverulegt) innrennsli útilokaðs árs og flokkar það
    með mörkum sem voru reiknuð án þess árs. Ekki falið innrennsli í þjálfun.
    """
    _leidbeining_stillingar(dV, dx)
    if not all(m in I_all.columns for m in range(1, 13)):
        raise ValueError('I_all verður að hafa dálka 1 til 12.')
    if not all(y in I_all.index for y in I_thj.index):
        raise ValueError('I_all vantar eitt eða fleiri prófunarár.')
    if 2024 in I_thj.index:
        raise ValueError('2024 má ekki vera í þjálfunargögnunum.')
    rows = []
    for ar in list(I_thj.index):
        train = I_thj.drop(index=ar)
        il_arn = cs.smida_innrennslislikan(train)
        st = cs.Stillingar(V1=float(V1), **_leidbeining_stillingar(dV, dx))
        lam, sd, mat, stada = cs.finna_lam(il_arn, st, krafa=krafa, matsfall='net')
        if sd is None:
            rows.append({'Prófunarár': ar, 'Þjálfunarár': len(train),
                         'Staða bestunar': stada, 'Áreiðanleiki þjálfunar (%)': 100 * mat['areidanleiki'],
                         'Forgangsskortur í prófun': np.nan, 'Skortsmanudir': np.nan,
                         'Afhent (GWh)': np.nan, 'Lokastaða (Mm³)': np.nan,
                         'Lagrange λ': lam})
            if framvinda:
                print(f'{ar}: engin stefna sem uppfyllir kröfuna samkvæmt líkaninu')
            continue
        I_ar = I_all.loc[ar, list(range(1, 13))].to_numpy(dtype=float)
        if not np.isfinite(I_ar).all():
            raise ValueError(f'Vantar innrennsli fyrir {ar}.')
        c_ar = [int(cs.flokkur(I_ar[t], t+1, il_arn.mork)) for t in range(12)]
        df = cs.keyra_ar(sd.stefnufall(), I_ar, c_ar, float(V1))
        s = cs.samantekt_ars(df, float(V1))
        rows.append({
            'Prófunarár': ar, 'Þjálfunarár': len(train), 'Staða bestunar': stada,
            'Áreiðanleiki þjálfunar (%)': 100 * mat['areidanleiki'],
            'Forgangsskortur í prófun': bool(s['skortsmanudir'] > 0),
            'Skortsmanudir': s['skortsmanudir'],
            'Afhent (GWh)': s['afhent_GWh'],
            'Skert (GWh)': s['skert_GWh'],
            'Lokastaða (Mm³)': s['V13'],
            'Yfirfall (Mm³)': s['yfirfall'],
            'Lagrange λ': lam,
        })
        if framvinda:
            print(f'{ar}: skortsmanudir={s["skortsmanudir"]}; '
                  f'afhent={s["afhent_GWh"]:.1f} GWh; '
                  f'áreiðanleiki þjálfunar={mat["areidanleiki"]:.1%}')
    return pd.DataFrame(rows).set_index('Prófunarár')


def breytt_thjalfun(I_thj: pd.DataFrame, grunnur, il, V1=1600,
                     ar_fra=2016, dV=5.0, dx=2.0, krafa=0.95):
    """Ein breytt forsenda: notum aðeins þjálfunarár >= ar_fra.

    Metum upprunalega frysta hópstefnu undir nýja líkindalíkaninu,
    endurleysum einnig undir nýju forsendunni og berum saman.
    Reiknum bæði með framvirkri dreifingu á neti (þó mismunandi netþéttleiki).
    """
    subset = I_thj.loc[I_thj.index >= ar_fra].copy()
    if len(subset) < 4:
        raise ValueError('Of fá þjálfunarár eftir breytingu.')
    nytt_il = cs.smida_innrennslislikan(subset)
    old = grunnur[V1]['sdp']
    fryst_old = cs.meta_net(old.stefna, il, V1, old.Vnet)
    fryst_new = cs.meta_net(old.stefna, il, V1, old.Vnet,
                            Ibar=nytt_il.Ibar, P=nytt_il.P, pi_des=nytt_il.pi_des)
    st = cs.Stillingar(V1=V1, **_leidbeining_stillingar(dV, dx))
    lam, sd, endur_mat, stada = cs.finna_lam(nytt_il, st, krafa=krafa, matsfall='net')
    rows = [
        {'Forsenda': '2015–2023, fryst hópstefna', 'Áreiðanleiki (%)': 100 * fryst_old['areidanleiki'],
         'Afhent (GWh)': fryst_old['E_afhent_GWh'], 'Fjöldi þjálfunarára': len(I_thj)},
        {'Forsenda': f'{ar_fra}–2023, fryst hópstefna', 'Áreiðanleiki (%)': 100 * fryst_new['areidanleiki'],
         'Afhent (GWh)': fryst_new['E_afhent_GWh'], 'Fjöldi þjálfunarára': len(subset)},
    ]
    if sd is not None:
        rows.append({'Forsenda': f'{ar_fra}–2023, endurleyst stefna',
                     'Áreiðanleiki (%)': 100 * endur_mat['areidanleiki'],
                     'Afhent (GWh)': endur_mat['E_afhent_GWh'],
                     'Fjöldi þjálfunarára': len(subset)})
    else:
        rows.append({'Forsenda': f'{ar_fra}–2023, engin gild stefna',
                     'Áreiðanleiki (%)': 100 * endur_mat['areidanleiki'],
                     'Afhent (GWh)': np.nan, 'Fjöldi þjálfunarára': len(subset)})
    return pd.DataFrame(rows).set_index('Forsenda'), nytt_il


def teikna_loyo(taflan):
    """Sýnir skortsár og afhenta orku, útilokuð ár á x-ás."""
    t = taflan.sort_index()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    y = t['Afhent (GWh)'].astype(float)
    axs[0].bar(t.index.astype(str), y)
    axs[0].set_ylabel('Afhent viðbótarorka (GWh)')
    axs[0].set_title('Afhending á útilokuðu ári')
    axs[0].tick_params(axis='x', rotation=45)
    z = t['Skortsmanudir'].astype(float)
    axs[1].bar(t.index.astype(str), z)
    axs[1].set_ylabel('Fjöldi mánaða með forgangsskorti')
    axs[1].set_title('Forgangsskortur á útilokuðu ári')
    axs[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    return fig


def samantekt_loyo(taflan):
    """Lýsir afmörkun athugunar; ekkert tölfræðilegt öryggisbil gefið í skyn."""
    gilt = taflan['Skortsmanudir'].dropna()
    return pd.Series({
        'Prófuð ár': len(gilt),
        'Ár með forgangsskorti': int((gilt > 0).sum()),
        'Hlutfall ára án skorts (%)': 100 * (gilt == 0).mean() if len(gilt) else np.nan,
        'Athugasemd': 'Þetta er árlegt LOYO-próf, ekki óháð staðfesting á 95% líkum.'
    })
