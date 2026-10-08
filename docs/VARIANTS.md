# LoRA variants — decision guide (2026 sweep summary)

| Variant | One-line idea | When to pick it | Cost |
|---|---|---|---|
| Vanilla LoRA (Hu'21) | $W_0+(\alpha/r)BA$, A gauss, B=0 | default; well-tuned matches most variants (2026 unified study) | 1× |
| QLoRA | 4-bit frozen base + LoRA-16bit | VRAM-bound (7B on 1 GPU) | slower steps |
| DoRA | magnitude + direction split | max quality, low r (needs 2 LRs) | +small |
| rsLoRA | scale $\alpha/\sqrt{r}$ | high r (32–256) stability | free |
| LoRA+ | $lr_B \gg lr_A$ | faster converge, B-from-zero stuck | free |
| PiSSA/OLoRA/EVA | SVD/principal init of A,B | hard domain shift, few steps | 1 SVD |
| AdaLoRA/GeLoRA/LR-LoRA | adaptive/learnable rank | mixed layer importance (cf. our H2/H3) | scheduler |
| MoE-LoRA/MiLoRA | routed multi-adapters | multi-task serving (S-LoRA) | router |

Rule from §7.1 + our H3: **more matrices at low r > one matrix at high r**.
Rule from 2026 scaling work: tune $\alpha$ first ($\alpha\approx2r$ start, up to $256\sqrt{r}$ for LLMs), LR second.
