# CONTRACT: KANJI PROJECT DOCUMENTATION & VERSIONING
# PRIORITY: CRITICAL
# ROLE: MLOps Agent

---

## CRITICAL BEHAVIORAL INSTRUCTIONS (META-RULES)

1.  **INCREMENTAL VERSIONING LOGIC**: Before generating the "ID" field in a [MODEL] commit, you must list the existing tags (`git tag -l`). Search for the pattern `<architecture>-model-vX`.
    * If none exist, X = 1.
    * If they exist (e.g., v1, v2), take the highest number and add 1 (e.g., v3).
    * Use this calculated ID for both the commit text and the subsequent `git tag` command.
2.  **HYBRID LANGUAGE**:
    * **Code and Commits**: ALWAYS in ENGLISH.
    * **Documentation (READMEs)**: ALWAYS in SPANISH.
3.  **DATA SOURCE HIERARCHY**: When extracting metrics (Accuracy, Loss, Hyperparameters, etc.) for commits, logs, READMEs or any sort of report:
    * **PRIORITY 1 (TRUTH)**: Notebook Output Cells. Always verify the actual printed values in the executed notebook. Take into consideration that some hyperparameters are defined in the first cell but then optimized with optuna.
    * **PRIORITY 2 (FALLBACK)**: Generated artifacts (e.g., `training_history.json`, `classes.json`) ONLY if the notebook output is cleared or ambiguous.
    * **NEVER**: Rely solely on file names or previous commit messages.

---

# 1. INVARIANTS (Global Constraints)
# ------------------------------------------------------------------
- CODE_LANGUAGE_INVARIANT: All code, variable names, and commit messages must be in ENGLISH.
- DOC_LANGUAGE_INVARIANT: All generated documentation (README.md files) must be in SPANISH.
- FORMAT_INVARIANT: No conversational filler. Output ONLY the requested artifact.
- INTEGRITY_INVARIANT: Do not hallucinate metrics. If data is missing for a [MODEL] commit, trigger a PRECONDITION_FAILURE error. Metrics (Val Acc, Loss) MUST correspond to the *BEST SAVED MODEL* (checkpoint), not necessarily the last training epoch.
- CHECKPOINT_INVARIANT: Never add a checkpoint to the git repository. It is not necessary since we are using git tags to version the models.

# 2. CONTRACT: COMMIT MESSAGES & TAGGING
# ------------------------------------------------------------------
# TRIGGER: User requests a commit.

## PRECONDITIONS
1. Changes must be staged (`git diff --staged`).
2. If tag is [MODEL], validation metrics (Val Acc, Val Loss) MUST be present in context.
3. **ID Calculation**: Scan existing tags to determine the next sequential version number (X) for `<arch>-model-vX`.
4. **TAG VALIDATION**: The [TAG] must be STRICTLY selected from this list:
   - `[MODEL]`: For saved model checkpoints (Triggers Auto-Tag).
   - `[ARCH]`: For network architecture changes.
   - `[DATA]`: For dataset/augmentation changes.
   - `[HYPER]`: For config/parameter tuning only.
   - `[FIX]`: For bug fixes.
   - `[FEAT]`: For new scripts/features.
   - `[DOCS]`: For documentation only.
   - `[REF]`: For bibliographic references.
   - `[LOG]`: For updates to `training_log.csv`.
   - `[MODIFY]`: For changes not covered by the other tags.



## POSTCONDITIONS (Strict Output Format)
```
[TAG] <Imperative Title in English>

<Description of changes in English>

--- EXPERIMENT SPECS ---
ID: <architecture>-model-v<next_sequential_number>
Arch: <Architecture Name>
Input: <Input Specs (e.g., 128x128 1ch or 3ch)>
Classes: <Num Classes>

--- HYPERPARAMETERS ---
Epochs: <Num>
Batch Size: <Num>
LR: <Num>
Optimizer: <Name>

--- FINAL RESULTS ---
Val Acc: <Value>%
Val Loss: <Value>
Test Acc: <Value>% (Optional)
Notes: <Observation in English>
```

### Bibliographic References Format ([REF])
```
[REF] <Short description of the reference in English>

File: <Filename or URL>
Author: <Author Name(s)>
Reference: <Full bibliographic reference or key info>
Rationale: <Brief explanation of why this was added in English>
```


## AUTOMATION HOOK (The "Auto-Tag" Rule)
AFTER successfully creating a commit with the [MODEL] tag:
1. Extract the "ID" field from the commit message (e.g., "resnet18-model-v3").
2. AUTOMATICALLY execute: `git tag -a <ID> -m "<Title> - Acc: <Val Acc>"`
3. Confirm to user: "Commit created and Tag <ID> applied."

# 3. CONTRACT: DIRECTORY READMEs
# ------------------------------------------------------------------
# TRIGGER: User requests "Generate README".

## REFERENCE STANDARD
- READ the file `train/README.md` first (if exists).
- MIMIC its structure, tone, and level of detail for the target directory.

## POSTCONDITIONS
The output must be a `README.md` (in SPANISH) following the reference structure:
- Descripción General (Overview)
- Tabla de Contenidos (File | Type | Description)
- Uso (Usage)
- Dependencias (Dependencies)

# 4. CONTRACT: AUTO-UPDATE DIRECTORY READMEs
# ------------------------------------------------------------------
# TRIGGER: User commits or adds files to a directory.

## PRECONDITIONS
1. A commit is being created that includes NEW or MODIFIED files in a directory.
2. The directory has an existing `README.md` file.

## POSTCONDITIONS
1. **Detect affected directories**: Identify which directories contain the staged files.
2. **Update README**: For each affected directory with a README.md:
   - Add new files to the "Contenido" (Contents) table if they are new.
   - Update descriptions if file purpose has changed.
   - Maintain the existing structure and Spanish language.
3. **Include in commit**: Stage the updated README.md files in the same commit.
4. **Notify user**: Confirm which READMEs were auto-updated.

## EXCEPTIONS
- Do NOT update README if only `.gitignore`, `.gitattributes`, or hidden files (starting with `.`) are modified.
- Do NOT update README for the root directory unless explicitly requested.
- Use `[LOG]` tag strictly for `training_log.csv` updates.

# 5. CONTRACT: HISTORIC PROTOCOL (LOG)
# ------------------------------------------------------------------
# TRIGGER: A commit with tag [MODEL] is executed.

## MANDATORY WORKFLOW
1.  **[MODEL] Commit**: Execute the commit with the `[MODEL]` tag and its corresponding Git Tag (e.g., `v0.1-resnet`).
2.  **Update CSV**: Register the results in `training_log.csv` (root of the project).
3.  **[LOG] Commit**: Add `training_log.csv` and execute a new commit with the `[LOG]` tag and a NEW Git Tag following the pattern `<experiment_id>-log`.

## CSV STRUCTURE (snake_case columns)
The file `training_log.csv` MUST maintain these columns strictly:
- `experiment_id`: (Must match Git Tag, e.g., v0.1-resnet)
- `timestamp`: (Format YYYY-MM-DD HH:MM)
- `architecture`: (e.g., ResNet18, MobileNetV3)
- `dataset_name`: (e.g., HSK-1, Custom-150)
- `classes`: (e.g., 55)
- `img_size`: (Input resolution)
- `batch_size`: (Hyperparameter)
- `learning_rate`: (Hyperparameter)
- `augmentation`: (Brief description, e.g., "Elastic+Erode")
- `epochs_run`: (Actual epochs executed)
- `val_accuracy`: (e.g., 90%)
- `val_loss`: (e.g., 5%)
- `test_accuracy`: (e.g., 70%) **CAUTION: Is on this print ""\nGlobal Test Accuracy: {acc*100:.2f}%".**
- `top_5_test`: (Top 5 accuracy on test set, e.g., 85%) **CAUTION: Is on this print ""Top-5 Accuracy: {top5_acc*100:.2f}%""**
- `top_5_casia`: (Top 5 accuracy on Casia dataset during training, e.g., 80%) **CAUTION: Is on this print "Top-5 Accuracy: <correct>/<total> (<acc>%)" under "====== Evaluation with CASIA train set ======" print output.**
- `train_loss_final`: (Final loss)
- `model_size_mb`: (Weight of generated .pth)
- `notes`: (Brief observations)
