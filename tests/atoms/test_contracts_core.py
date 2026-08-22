"""Contracts for the everywhere-atoms: svd, qr, cholesky, lstsq, pinv, fft.

Each law is checked twice: the true result passes, a corrupted result
fails. A contract that cannot fail is not a check.
"""

import numpy as np

from skverify.contracts import FAILED, OK, check_call

rng = np.random.default_rng(5)
A = rng.uniform(0.5, 2.0, (4, 3))
S = A.T @ A + 0.5 * np.eye(3)
b = rng.uniform(0.5, 2.0, 4)
v = rng.uniform(0.5, 2.0, 8)


def verdict(name, args, result):
    return dict(check_call(name, args, result)[1])["residual"]


def test_svd_law():
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    assert verdict("svd", [A], (u, s, vh)) == OK
    assert verdict("svd", [A], (u, s * 1.01, vh)) == FAILED


def test_qr_law():
    q, r = np.linalg.qr(A)
    assert verdict("qr", [A], (q, r)) == OK
    assert verdict("qr", [A], (q, r + 0.01)) == FAILED


def test_cholesky_law():
    ell = np.linalg.cholesky(S)
    assert verdict("cholesky", [S], ell) == OK
    assert verdict("cholesky", [S], ell * 1.01) == FAILED


def test_lstsq_law():
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    assert verdict("lstsq", [A, b], (x,)) == OK
    assert verdict("lstsq", [A, b], (x + 0.05,)) == FAILED


def test_pinv_law():
    x = np.linalg.pinv(A)
    assert verdict("pinv", [A], x) == OK
    assert verdict("pinv", [A], x * 1.01) == FAILED


def test_fft_laws():
    spec = np.fft.fft(v)
    assert verdict("fft", [v], spec) == OK
    assert verdict("fft", [v], spec * 1.001) == FAILED
    assert verdict("ifft", [spec], np.fft.ifft(spec)) == OK
    rspec = np.fft.rfft(v)
    assert verdict("rfft", [v], rspec) == OK
    assert verdict("irfft", [rspec], np.fft.irfft(rspec)) == OK
    assert verdict("irfft", [rspec], np.fft.irfft(rspec) + 0.01) == FAILED
