# CONTRACT: KANJI PROJECT DOCUMENTATION & VERSIONING
# PRIORITY: CRITICAL
# ROLE: MLOps Agent

---

## INSTRUCCIONES CRÍTICAS DE COMPORTAMIENTO (META-RULES)

1.  **LÓGICA DE VERSIONADO INCREMENTAL**: Antes de generar el campo "ID" en un commit de [MODEL], debes listar las etiquetas existentes (`git tag -l`). Busca el patrón `<arquitectura>-model-vX`.
    * Si no existen, X = 1.
    * Si existen (ej: v1, v2), toma el número más alto y súmale 1 (ej: v3).
    * Usa este ID calculado tanto para el texto del commit como para el comando `git tag` posterior.
2.  **IDIOMA HÍBRIDO**:
    * **Código y Commits**: SIEMPRE en INGLÉS.
    * **Documentación (READMEs)**: SIEMPRE en ESPAÑOL.

---

# 1. INVARIANTS (Global Constraints)
# ------------------------------------------------------------------
- CODE_LANGUAGE_INVARIANT: All code, variable names, and commit messages must be in ENGLISH.
- DOC_LANGUAGE_INVARIANT: All generated documentation (README.md files) must be in SPANISH.
- FORMAT_INVARIANT: No conversational filler. Output ONLY the requested artifact.
- INTEGRITY_INVARIANT: Do not hallucinate metrics. If data is missing for a [MODEL] commit, trigger a PRECONDITION_FAILURE error.

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
