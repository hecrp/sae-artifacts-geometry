# Local PCA decoder/encoder neighborhoods for H7

Status: no-training vector-based diagnostic using public Bloom GPT-2 Small L8 SAE. Conservative: not proof of H7.

## Metadata
- release: gpt2-small-res-jb
- sae_id: blocks.8.hook_resid_pre
- W_enc_shape: [768, 24576]
- W_dec_shape: [24576, 768]
- status: no SAE training; public pretrained SAE loaded for local geometry diagnostics
- global_cos_mean: 0.36828580498695374
- global_cos_std: 0.07853515446186066

## Group-level PCA
| Group | n | Decoder PR | Decoder PC2/PC1 | Decoder rank95 | Encoder PR | Encoder PC2/PC1 | Mean decoder cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| hedge_family | 4 | 1.9372193288257102 | 0.4720618769884831 | 2 | 1.887945182465877 | 0.5591363462765022 | 0.05739471781998873 |
| parent_cluster | 3 | 1.2959437451788138 | 0.1513619743931011 | 2 | 1.5989993574991923 | 0.33263891296254394 | 0.7683761517206827 |
| random_l8 | 4 | 2.9601831594629076 | 0.8717336573087154 | 3 | 2.6952965886255957 | 0.6087015555888724 | 0.014236619773631295 |
| controls | 3 | 1.9949807313836414 | 0.9044730948215159 | 2 | 1.9957821484142122 | 0.912097908878901 | -0.0113296111424764 |
| hedge_plus_parent | 7 | 2.22571545207278 | 0.363307020567235 | 3 | 3.4948578927319716 | 0.5258393225449426 | -0.13050471028933922 |
| all_seed_groups | 14 | 6.415883299239346 | 0.35832511524596716 | 9 | 3.844401270839193 | 0.26210274163374503 | -0.031152743400985394 |

## Seed-neighborhood PCA
| Feature | Group | n nodes | Decoder PR | Decoder PC2/PC1 | Encoder PR | Top nodes |
|---:|---|---:|---:|---:|---:|---|
| F19139 | hedge_family | 11 | 7.086947061730751 | 0.6739930373507594 | 6.178470191641337 | [19139, 11969, 21829, 21714, 4872, 3441, 23881, 1195] |
| F18573 | hedge_family | 7 | 4.639014611061308 | 0.6889010530695131 | 1.9682445989150976 | [18573, 16152, 21174, 8488, 5970, 8683, 11973] |
| F11969 | hedge_family | 11 | 7.083839639402111 | 0.6528126775388958 | 5.012720700162305 | [11969, 19139, 15019, 7145, 16114, 3787, 7421, 2508] |
| F20804 | hedge_family | 8 | 6.594415569301046 | 0.7486822667111219 | 6.565043823676278 | [20804, 8341, 4061, 4012, 7406, 3781, 13850, 20808] |
| F17487 | parent_cluster | 3 | 1.2959437451788138 | 0.1513619743931011 | 1.5989993574991923 | [17487, 21174, 16152] |
| F21174 | parent_cluster | 4 | 1.380196822016173 | 0.1517712625198788 | 1.8766611267426376 | [21174, 16152, 17487, 18573] |
| F16152 | parent_cluster | 4 | 1.3801968220161729 | 0.15177126251987866 | 1.8766611267426387 | [16152, 21174, 17487, 18573] |
| F22561 | random_l8 | 11 | 8.681132963892933 | 0.8573036716632157 | 9.4800185287526 | [22561, 15944, 9152, 12047, 8233, 16579, 13526, 3990] |
| F6165 | random_l8 | 11 | 8.10171069514796 | 0.7395267237767981 | 7.857478977662602 | [6165, 13600, 5245, 3095, 15359, 23054, 23449, 16115] |
| F5977 | random_l8 | 11 | 8.124242238055713 | 0.7033375122643418 | 9.518052651891706 | [5977, 6674, 24049, 17285, 20546, 11176, 4316, 11425] |
| F3696 | random_l8 | 11 | 7.433214971623945 | 0.5741302410025255 | 9.350768178576292 | [3696, 6853, 5517, 15971, 23955, 21026, 5480, 5744] |
| F1803 | controls | 11 | 6.9010647586656555 | 0.6874923246701992 | 9.062826761440201 | [1803, 14355, 23746, 23638, 7520, 10687, 13470, 13576] |
| F1442 | controls | 11 | 7.164386469881864 | 0.7629581921267872 | 8.5951515382617 | [1442, 12080, 21625, 19490, 9101, 2281, 5880, 18610] |
| F10666 | controls | 11 | 6.973102641959215 | 0.5059131535409049 | 6.970778570808223 | [10666, 11978, 22337, 3959, 16269, 9837, 15275, 18462] |

## Tangent/normal decomposition relative to local decoder PCA
| Feature | available | E_parallel_frac | D_parallel_frac | n_tangent |
|---:|---|---:|---:|---:|
| F19139 | True | 0.24145782938936816 | 0.49890463091828613 | 2 |
| F18573 | True | 0.5867049510506865 | 0.09439516555398504 | 2 |
| F11969 | True | 0.4001403793899362 | 0.8105402273145583 | 2 |
| F20804 | True | 0.021285538029067033 | 0.15042392606450009 | 2 |
| F17487 | True | 0.3840412993711576 | 0.40929923051036937 | 2 |
| F21174 | True | 0.7179268186445001 | 0.9460311783831467 | 2 |
| F16152 | True | 0.7173344997670217 | 0.9160671234130859 | 2 |
| F22561 | True | 0.08493392213145706 | 0.042698626706581214 | 2 |
| F6165 | True | 0.0711448632063366 | 0.09056386200369171 | 2 |
| F5977 | True | 0.10265460691724385 | 0.2012835674592386 | 2 |
| F3696 | True | 0.00881073778727341 | 0.11106496655778522 | 2 |
| F1803 | True | 0.056716782982622595 | 0.13463524707951743 | 2 |
| F1442 | True | 0.0886294809928956 | 0.28729779299572006 | 2 |
| F10666 | True | 0.028250648585587745 | 0.1469408357092118 | 2 |

## Actual W_dec pair cosines for Supplement 4 hedge-parent pairs
| Pair | Supplement 4 reported (withdrawn March 2026 version) | Actual W_dec cosine | abs diff |
|---|---:|---:|---:|
| F19139-F21174 | -0.705 | -0.9900655746459961 | 0.28506557464599613 |
| F11969-F16152 | -0.48 | -0.9255034923553467 | 0.4455034923553467 |
| F11969-F21174 | -0.312 | -0.8632187843322754 | 0.5512187843322753 |
| F19139-F16152 | -0.198 | -0.9078973531723022 | 0.7098973531723023 |
| F19139-F17487 | -0.105 | -0.7046343088150024 | 0.5996343088150025 |
| F11969-F17487 | -0.091 | -0.4799840450286865 | 0.38898404502868655 |

## Conservative conclusion
Vector-based local PCA is now available for the public GPT-2 L8 SAE. The hedge+parent decoder set has participation_ratio=2.226 and PC2/PC1=0.363; the parent cluster has participation_ratio=1.296. These values should be interpreted as local geometric diagnostics, not proof of H7. They can strengthen or weaken H7 only after comparison to matched controls and after checking whether the same structure persists under broader neighborhoods.

## Interpretation discipline
- Observation: local PCA/PR values are direct calculations on public SAE vectors.
- Inference: values >1 indicate local dimensionality beyond a single line, but small-n PCA is noisy.
- Hypothesis: if anti-aligned neighborhoods are consistently higher-dimensional than controls, this supports H7.
- Not claimed: this run alone does not demonstrate manifold fragmentation or SAE shattering.