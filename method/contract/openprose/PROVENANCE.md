# Vendored open-prose contracts & fixtures — provenance

- **Source repo:** open-prose (polyrepo sibling; dev-only reference, never a runtime path)
- **Source commit:** a0395cdb004fb1782dd45e145f24f948e61043d1
- **Vendored:** 2026-07-16
- **Pattern:** pinned copy inside the consumer (workspace rule; same as the RD-007
  learning-event vendoring in `method/contract/learning-event-v1.schema.json`)
- **Contents:** `contracts/{receipt.md,ir.md}`; corpus runs from
  `skills/prose/examples/runs/`; broken run fixtures from `tests/fixtures/runs/`
  (each with its upstream `expected.json`); IR fixtures from `tests/fixtures/ir/`
  (vendored for the future IR reader — NOT consumed by any v1 test).
- **Excluded:** fixture `generate.py` scripts (they import `openprose_tools`,
  which is not vendored).
- **Contract policy:** append-frozen — unknown fields are ignored (but hashed as
  received), unknown `v` values are refused.
- **Refresh:** manual — re-copy from a new upstream commit, re-run
  `find … -exec shasum -a 256 {} \; | sort -k2`, update this file. Automated
  byte-conformance is a follow-up (same posture as RD-007 M2).

## sha256 (at vendoring time)

```
cb3bee648fcde711083590403b401fc2c70e689f000ad67b787a36c807972614  method/contract/fixtures/openprose/broken/broken-chain/expected.json
6c7932dc9944af99eeca663fd55c7bb17fc5c4f3746160adeb6706cae4430562  method/contract/fixtures/openprose/broken/broken-chain/receipts.jsonl
7353014f3703593ec6e489476e76595ced969cd6e497ed1118848e6f9c02b922  method/contract/fixtures/openprose/broken/broken-chain/run.json
885f8316ca55a1e8f435d1e9801fa3be23cdd1d94541402f1116692f4edf1f96  method/contract/fixtures/openprose/broken/tampered-content/expected.json
7df719580641bb489e92759ab277ed796bae37416e02b558cfd1fcee9589b5fe  method/contract/fixtures/openprose/broken/tampered-content/receipts.jsonl
cfab40a28ae07d4c92281551b0c17f95ec00d4dc2006b21ef8184fc7d770b0f9  method/contract/fixtures/openprose/broken/tampered-content/run.json
b52654e7cb1adc077a57d58e20d9c3d713183adb7e6ac3711f1b4ed54166e21b  method/contract/fixtures/openprose/broken/torn-write/expected.json
586a676a22dab9770afe0e2bc59647899ece1aa078950c80499aae1e68623d15  method/contract/fixtures/openprose/broken/torn-write/receipts.jsonl
8b06ff6b239b26f3ffee88b4ee99b4f11819b2a303693b49b77056ace02c4fa5  method/contract/fixtures/openprose/broken/torn-write/run.json
4ab05d2ed21767abf8bcfb8dc15ac64b69ef7fc5eb83369a503adfdf45d98a1b  method/contract/fixtures/openprose/broken/truncated-ledger/expected.json
586a676a22dab9770afe0e2bc59647899ece1aa078950c80499aae1e68623d15  method/contract/fixtures/openprose/broken/truncated-ledger/receipts.jsonl
cfab40a28ae07d4c92281551b0c17f95ec00d4dc2006b21ef8184fc7d770b0f9  method/contract/fixtures/openprose/broken/truncated-ledger/run.json
fbfc89c51d317c3ed6796de7077c6daf9f6ac5cfb91fa0325c602891433f1236  method/contract/fixtures/openprose/ir/stale-source/dist/program.ir.json
080b7500d4aefef23fe9d1ab79f8ee5a04b88b8fb56aa6e9d2eb737dd859fb28  method/contract/fixtures/openprose/ir/stale-source/expected.json
3a8d49c9573b0fc13ba37e954d7c60e7551a64e74eeb39f94c2bd14d415bb692  method/contract/fixtures/openprose/ir/stale-source/program.prose
fd9ae2136d5b12d3186dd9b9a10471e2392bf2c8dad52fb9f99dfee4e171bd03  method/contract/fixtures/openprose/ir/tampered-ir/dist/program.ir.json
885f8316ca55a1e8f435d1e9801fa3be23cdd1d94541402f1116692f4edf1f96  method/contract/fixtures/openprose/ir/tampered-ir/expected.json
3a8d49c9573b0fc13ba37e954d7c60e7551a64e74eeb39f94c2bd14d415bb692  method/contract/fixtures/openprose/ir/tampered-ir/program.prose
3c3f9892aec3c5958a2a01f4192dad5c1b495d2451d2fcb5cf491fed2a545a90  method/contract/fixtures/openprose/ir/unknown-agent/dist/program.ir.json
21e1898d1b19103dc017ff37db07d1b7b77671ec451f920dcbc4b25d6d608b21  method/contract/fixtures/openprose/ir/unknown-agent/expected.json
3a8d49c9573b0fc13ba37e954d7c60e7551a64e74eeb39f94c2bd14d415bb692  method/contract/fixtures/openprose/ir/unknown-agent/program.prose
45c5f28d41209fef8256e7970c15c7d44962001569b1ccb65a72a47b90cad611  method/contract/fixtures/openprose/runs/20260716-082519-hak1oi/bindings/anon_001.md
ec64fc6b72e6eb331ac567e9f95a6fea24e8938c85a5ed0ff6282a6bfeb0f0fc  method/contract/fixtures/openprose/runs/20260716-082519-hak1oi/program.prose
75d0cd0cbcc8fa4412a98bf2337c6eeddc51f373cf00debfdf820b19cb6257bf  method/contract/fixtures/openprose/runs/20260716-082519-hak1oi/receipts.jsonl
9bbd30911ea113f632eece3e32ccdd175f8e0d59da939255077bea398373e03f  method/contract/fixtures/openprose/runs/20260716-082519-hak1oi/run.json
e548d9ffa44cd8b6723491ed18bc333ff9235e801853d201227fc74899904f3e  method/contract/fixtures/openprose/runs/20260716-082519-hak1oi/state.md
b73fc342800377494d43c5e40648870d37ecd605297f1fca9d7112d087af2133  method/contract/fixtures/openprose/runs/20260716-082734-0vx3bm/bindings/anon_006.md
1142d37cc7c1a219f6accbda339e75b3801a8740dc9bcbf2ed84fad187ec1d7c  method/contract/fixtures/openprose/runs/20260716-082734-0vx3bm/bindings/perf.md
3f6ddc7a0e3e5e817e0bb069b8e6837b662583286c34625affdf2466d9fb7a85  method/contract/fixtures/openprose/runs/20260716-082734-0vx3bm/bindings/security.md
e47d563dbad65c6086ebbe02415b96b6eed5211cf12f5d348688bed494f14b82  method/contract/fixtures/openprose/runs/20260716-082734-0vx3bm/bindings/style.md
cdce5073a4f3fe4aaa871cb9ee3a7890076fe1561ba77221fba91df3567796d1  method/contract/fixtures/openprose/runs/20260716-082734-0vx3bm/program.prose
fc11d03ab3ca91570231fefcf59eb2a265cb79999030519c174b4af9878f7155  method/contract/fixtures/openprose/runs/20260716-082734-0vx3bm/receipts.jsonl
dbfd0211295baa6a11353034247871771c87e091ea431b269d9eaeedadb01ded  method/contract/fixtures/openprose/runs/20260716-082734-0vx3bm/run.json
e03c667fc1a4269bb2606387d3e732025d31cf6acf3bcf68911be38c543006e2  method/contract/fixtures/openprose/runs/20260716-082734-0vx3bm/state.md
627d99516ac696328510446baa5c06d50af7671b07a564b887ef013d8f06fee3  method/contract/fixtures/openprose/runs/20260716-083117-6pgtu7/bindings/plan.md
e92b0df8516fe556ee32a969be6df49277530d0456c7c68cd93e255838909ba9  method/contract/fixtures/openprose/runs/20260716-083117-6pgtu7/bindings/review.md
eedec498942b01d24ad9a4d5179b0747ff48ec8f375e2c4141f79e31e903bcfa  method/contract/fixtures/openprose/runs/20260716-083117-6pgtu7/bindings/synthesis.md
c335e30430b4fab15e41401f8627012b9d56e59d0fb9cfd9b3f98d3e2a69f510  method/contract/fixtures/openprose/runs/20260716-083117-6pgtu7/bindings/task.md
b37aa7a70ff0aea250cd6f0001416eea2d001298eca255ffb9d3e9c58f66afe2  method/contract/fixtures/openprose/runs/20260716-083117-6pgtu7/bindings/work.md
b40a0121088e8d181a3324db442a651ba6b47c5e535f3c4a229c57501f4f527c  method/contract/fixtures/openprose/runs/20260716-083117-6pgtu7/program.prose
43a7462f8c2fa3980e5ef5be0a8a3e28f6db61f64e2a2c569e4a747f30390dd1  method/contract/fixtures/openprose/runs/20260716-083117-6pgtu7/receipts.jsonl
c025abf4dcafb6b59bbd1a7237e01108198e9cb361e34a1454e7b59be5c494a3  method/contract/fixtures/openprose/runs/20260716-083117-6pgtu7/run.json
6f800d97ad8d230c866eb406dc169237430dff1db18e2deab14f6073320da143  method/contract/fixtures/openprose/runs/20260716-083117-6pgtu7/state.md
b73fc342800377494d43c5e40648870d37ecd605297f1fca9d7112d087af2133  method/contract/fixtures/openprose/runs/20260716-094019-fsylki/bindings/anon_006.md
1142d37cc7c1a219f6accbda339e75b3801a8740dc9bcbf2ed84fad187ec1d7c  method/contract/fixtures/openprose/runs/20260716-094019-fsylki/bindings/perf.md
3f6ddc7a0e3e5e817e0bb069b8e6837b662583286c34625affdf2466d9fb7a85  method/contract/fixtures/openprose/runs/20260716-094019-fsylki/bindings/security.md
e47d563dbad65c6086ebbe02415b96b6eed5211cf12f5d348688bed494f14b82  method/contract/fixtures/openprose/runs/20260716-094019-fsylki/bindings/style.md
cdce5073a4f3fe4aaa871cb9ee3a7890076fe1561ba77221fba91df3567796d1  method/contract/fixtures/openprose/runs/20260716-094019-fsylki/program.prose
04b360f322d95c363a0bcd7845e1676d82a2fae1561a496c8f585e86c73f8ab9  method/contract/fixtures/openprose/runs/20260716-094019-fsylki/receipts.jsonl
e110e3bb39387c7347b38e6307e198d7b7eb60ab5c34935baf23447a5fff6978  method/contract/fixtures/openprose/runs/20260716-094019-fsylki/run.json
ed90107c03ced757bec047b9e436d90cbdc6544289a374c0f57c4d6c06dd226a  method/contract/fixtures/openprose/runs/20260716-094019-fsylki/state.md
29b308be85b85ffd3d3d50c954efe26f7864594600be5ac148385423f1028dd7  method/contract/openprose/ir.md
fceb323c27c817907d990ed59b203ffea6bbac3124168e4642fa238062e55306  method/contract/openprose/receipt.md
```
