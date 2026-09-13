# Source attribution

The initial `go-core`, `go-search` and `go-actors` crates are copied from the
user-provided `/home/cubic27/go` working tree, including their tests. That tree
declares MIT in its Rust workspace. Exact copied-file SHA-256 hashes are in
[provenance/go-source.json](provenance/go-source.json); no Git identity was
available for that source folder.

The Python binding adapts `crates/go-bridge`; the typed wrapper adapts
`gozero/native.py`, and the transport adapts `gozero/gtp.py`. Native-interface
tests and the KataGo qualifier draw on `tests/test_native.py` and
`eval/qualify_katago.py`. These reference hashes, the scoring fixture, GTP tests,
KataGo receipts and oracle helper are recorded in the same manifest. The new
package does not import the previous research framework at runtime.

KataGo-derived scoring code retains
[its license](crates/go-core/KATAGO_LICENSE.md). The Gumbel search implementation
retains the [Mctx Apache 2.0 license](crates/go-search/MCTX_LICENSE.md) and its
source comments. `go-search` declares `MIT AND Apache-2.0`.
