import matplotlib.pyplot as plt
import numpy as np
import pycwt as wavelet


t0 = 0
dt = 1/26.  # in years

mother = wavelet.Morlet(6)
s0 = 2 * dt  # Starting scale, in this case 2 * 1/26 years = 28 days
dj = 1 / 12  # Twelve sub-octaves per octaves
J = 7 / dj  # Seven powers of two with dj sub-octaves


def norm_data(dat):

    N = dat.size
    t = np.arange(0, N) * dt + t0

    p = np.polyfit(t - t0, dat, 1)
    dat_notrend = dat - np.polyval(p, t - t0)
    std = dat_notrend.std()  # Standard deviation
    dat_norm = dat_notrend / std  # Normalized dataset

    return dat_norm, std


def get_wavelet_spectrum(dat):

    dat_norm, _ = norm_data(dat)
    return wavelet.cwt(dat_norm, dt, dj, s0, J, mother)


def get_max_wavelet_power(dat):

    wave, _, freqs, _, _, _ = get_wavelet_spectrum(dat)

    power = (np.abs(wave)) ** 2
    glbl_power = power.mean(axis=1)
    period = 1 / freqs
    
    return period[np.argmax(glbl_power)]


def get_phase_diffs(dat, ref_ix):
    
    ref_wave, _, freqs, _, _, _ = get_wavelet_spectrum(dat[:, ref_ix])
    period = 1 / freqs

    phase_diffs = np.zeros(dat.shape[1])

    for i in range(dat.shape[1]):

        if i == ref_ix:
            continue

        wave, _, _, _, _, _ = get_wavelet_spectrum(dat[:, i])

        cross_power = wave * ref_wave.conj()

        # TODO: expose filters as arguments, double-check math
        ind = np.where(np.logical_and(period < 2.5, period > 1.5))
        phase_diffs[i] = np.angle(np.mean(cross_power[ind, 0*26:2*26]))
        
    return phase_diffs


def plot_wavelet_spectrum(dat):

    N = dat.size
    t = np.arange(0, N) * dt + t0

    dat_norm, std = norm_data(dat)
    var = std ** 2  # Variance
    alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise

    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J, mother)
    iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std

    power = (np.abs(wave)) ** 2
    fft_power = np.abs(fft) ** 2
    period = 1 / freqs

    signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                            significance_level=0.95,
                                            wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95

    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                            significance_level=0.95, dof=dof,
                                            wavelet=mother)

    fig, bx = plt.subplots(1, 1, figsize=(8, 3))
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    bx.contourf(t, np.log2(period), np.log2(power), np.log2(levels),
                extend='both', cmap=plt.cm.viridis)
    extent = [t.min(), t.max(), 0, max(period)]
    bx.contour(t, np.log2(period), sig95, [-99, 1], colors='k', linewidths=2,
            extent=extent)
    bx.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,
                            t[:1] - dt, t[:1] - dt]),
            np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),
                            np.log2(period[-1:]), [1e-9]]),
            'k', alpha=0.3, hatch='x')
    # bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(label, mother.name))
    # bx.set_ylabel('Period (years)')
    Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                            np.ceil(np.log2(period.max())))
    bx.set_yticks(np.log2(Yticks))
    bx.set_yticklabels(Yticks)
    bx.set(ylim=(np.log2(0.125), np.log2(8)))
    fig.set_tight_layout(True)


    fig, cx = plt.subplots(1, 1)
    cx.plot(glbl_signif, np.log2(period), 'k--')
    cx.plot(var * fft_theor, np.log2(period), '--', color='#cccccc')
    cx.plot(var * fft_power, np.log2(1./fftfreqs), '-', color='#cccccc', linewidth=1.)
    cx.plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
    cx.set_title('Global Wavelet Spectrum')
    # cx.set_xlabel(r'Power [({})^2]'.format(units))
    cx.set_xlim([0, glbl_power.max() * var])
    cx.set_ylim(np.log2([period.min(), period.max()]))
    cx.set_yticks(np.log2(Yticks))
    cx.set_yticklabels(Yticks)

    # print(period[np.argmax(glbl_power)])