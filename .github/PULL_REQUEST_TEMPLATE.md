# Summary

<!-- 这个 PR 做了什么？解决了什么问题？ -->

## Type of change

- [ ] Bug fix
- [ ] New feature (CSR-Faith / CIT-Faith)
- [ ] Refactor / cleanup
- [ ] Docs / scripts
- [ ] Other:

## Checklist

- [ ] `python -m unittest discover -s tests` 全部通过
- [ ] `python -m py_compile $(find verl scripts tests -name "*.py")` 无报错
- [ ] `bash -n scripts/*.sh scripts/extras/*.sh` 无报错
- [ ] **未提交**数据集 / 模型权重 / checkpoint / 缓存 / `scripts/env.local.sh`（代码与数据分离）
- [ ] 如改动训练脚本，已同步更新 `docs/csrfaith_run_guide.md` 与 `tests/test_training_scripts.py`

## Notes for reviewers

<!-- 需要 reviewer 重点关注的地方、已知限制、后续 TODO 等 -->
