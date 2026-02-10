# Referencias Bibliográficas para TFG - Clasificación de Kanji

Este documento identifica todos los conceptos, técnicas y herramientas utilizadas en el proyecto que requieren referencias académicas o bibliográficas para la memoria del TFG.

---

## 1. Datasets Utilizados

### ETL9B / ETL9G
**Qué es:** Dataset de caracteres Kanji manuscritos del Instituto de Investigación Electrotécnica de Japón.

**Cuándo se usó:**
- Usado como dataset principal de entrenamiento desde `resnet18-model-v1` hasta `ghostnet-model-v3`
- Mencionado en [training_log.csv](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/training_log.csv)

**Referencia sugerida:**
> ETL Character Database, Electrotechnical Laboratory, AIST, Japan

---

### CASIA Online and Offline Chinese Handwriting Databases
**Qué es:** Dataset de caracteres chinos manuscritos usado para evaluación externa (distribución diferente).

**Cuándo se usó:**
- Usado como dataset de evaluación externa para métricas `top_5_casia` 
- Visible en la carpeta `DATA/chinese-handwriting`
- Mencionado en múltiples commits del log

**Referencia sugerida:**
> Liu, C. L., Yin, F., Wang, D. H., & Wang, Q. F. (2011). CASIA online and offline Chinese handwriting databases. *ICDAR 2011*

---

### KanjiVG
**Qué es:** Base de datos vectorial de Kanji con información de componentes (radicals/strokes).

**Cuándo se usó:**
- Usado para obtener los componentes de cada Kanji para la arquitectura Multi-Head (commits a partir de `27001c5`)
- Archivo JSON con componentes en [dataset.py](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/dataset.py#L21-24)

**Referencia sugerida:**
> KanjiVG Project. https://kanjivg.tagaini.net/

---

## 2. Arquitecturas de Redes Neuronales

### ResNet (Residual Networks)
**Qué es:** Arquitectura con conexiones residuales que permite entrenar redes muy profundas.

**Cuándo se usó:**
- Experimentos `resnet18-model-v1` a `resnet18-model-v4`
- Primeros modelos del proyecto

**Referencia sugerida:**
> He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*

---

### MobileNetV3
**Qué es:** Arquitectura eficiente diseñada para dispositivos móviles con búsqueda de arquitectura automatizada.

**Cuándo se usó:**
- Experimentos `mobilenet_v3-model-v1` a `mobilenet_v3-model-v4`
- Usado en [models.py](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/models.py#L6-11) como backbone principal

**Referencia sugerida:**
> Howard, A., et al. (2019). Searching for MobileNetV3. *ICCV 2019*

---

### GhostNet
**Qué es:** Red eficiente que genera "feature maps fantasma" con operaciones lineales baratas.

**Cuándo se usó:**
- Experimentos `ghostnet-model-v1` a `ghostnet-model-v3`
- Usado como backbone en arquitectura Multi-Head (commit `1b96518`)
- [models.py](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/models.py#L24): `timm.create_model('ghostnet_100')`

**Referencia sugerida:**
> Han, K., et al. (2020). GhostNet: More Features from Cheap Operations. *CVPR 2020*

---

### MobileViT
**Qué es:** Arquitectura híbrida que combina convoluciones con Transformers para clasificación eficiente.

**Cuándo se usó:**
- Experimento `mobilevitv2-model-v1` (commit `416f0dc`)

**Referencia sugerida:**
> Mehta, S., & Rastegari, M. (2022). MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer. *ICLR 2022*

---

### FastViT
**Qué es:** Vision Transformer eficiente con bloques de atención rápidos.

**Cuándo se usó:**
- Experimentos `mobilenet_v3_fastvit-model-v1` y `v2` (arquitectura híbrida)

**Referencia sugerida:**
> Vasu, P. K. A., et al. (2023). FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization. *ICCV 2023*

---

### CRNN (CNN + RNN)
**Qué es:** Arquitectura que combina capas convolucionales con recurrentes.

**Cuándo se usó:**
- Experimentos `crnn-model-v1` y `crnn-model-v2`
- Commit `5f421bb` menciona replicación de arquitectura de Kaggle

**Referencia sugerida:**
> Shi, B., Bai, X., & Yao, C. (2017). An End-to-End Trainable Neural Network for Image-based Sequence Recognition. *IEEE TPAMI*

---

### Multi-Head Architecture (Arquitectura propia)
**Qué es:** Arquitectura con dos cabezas: una para clasificar componentes y otra para el Kanji, usando información composicional.

**Cuándo se usó:**
- Introducido en commit `27001c5` y refinado en `ccddafe`, `c754ad4`
- Definido en [models.py](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/models.py#L18-41)

**Referencia sugerida:** 
Esta es una **contribución propia**, pero puedes citar trabajos relacionados sobre:
- Multi-task learning
- Compositional character recognition

> Caruana, R. (1997). Multitask Learning. *Machine Learning, 28(1)*

---

## 3. Técnicas de Data Augmentation

### Transformaciones Geométricas

#### Random Perspective Transform
**Cuándo se usó:** Desde los primeros modelos, en [transforms.py](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/transforms.py#L40)

**Referencia sugerida:**
> Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003). Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis. *ICDAR 2003*

---

#### Random Affine Transform
**Cuándo se usó:** En todos los modelos ([transforms.py L41-46](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/transforms.py#L41-46))

**Referencia sugerida:**
(Misma que Perspective - Simard et al., 2003)

---

#### Elastic Transform
**Qué es:** Deformación elástica que simula variabilidad natural en escritura manuscrita.

**Cuándo se usó:** En todos los modelos ([transforms.py L47](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/transforms.py#L47))

**Referencia sugerida:**
> Simard, P. Y., et al. (2003). Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis. *ICDAR 2003*

---

### Transformaciones Morfológicas (Erode/Dilate)
**Qué es:** Operaciones de morfología matemática para simular variación en grosor de trazos.

**Cuándo se usó:** 
- Implementado en [transforms.py L18-34](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/transforms.py#L18-34)
- Mencionado en commit `3444174` como causante de problemas (se ajustó)

**Referencia sugerida:**
> Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.

---

### Random Erasing
**Qué es:** Augmentation que borra aleatoriamente rectángulos de la imagen.

**Cuándo se usó:** [transforms.py L54](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/transforms.py#L54)

**Referencia sugerida:**
> Zhong, Z., et al. (2020). Random Erasing Data Augmentation. *AAAI 2020*

---

### Gaussian Blur
**Cuándo se usó:** [transforms.py L48](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/transforms.py#L48)

(Técnica estándar de procesamiento de imágenes, referencia opcional)

---

### ColorJitter / GaussianNoise (REMOVIDOS)
**Cuándo se usó y quitó:**
- Usados inicialmente
- Removidos en commit `3444174` porque causaban overfitting en datos binarizados
- Documentado en [README.md](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/README.md#L40-55)

**Referencia sugerida:** Puedes documentar esto como **lección aprendida** sobre incompatibilidad de augmentations con dominios específicos.

---

## 4. Preprocesamiento de Imágenes

### Binarización con Otsu
**Qué es:** Método de umbralización automática adaptativa.

**Cuándo se usó:** [image_processing.py L9-13](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/image_processing.py#L9-13)

**Referencia sugerida:**
> Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Transactions on Systems, Man, and Cybernetics*

---

## 5. Técnicas de Optimización

### Transfer Learning (Pre-trained Models)
**Qué es:** Usar modelos pre-entrenados en ImageNet como punto de partida.

**Cuándo se usó:**
- En todos los modelos: `weights=models.MobileNet_V3_Large_Weights.DEFAULT`
- [models.py L7](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/models.py#L7), L24 (`pretrained=True`)

**Referencia sugerida:**
> Yosinski, J., et al. (2014). How transferable are features in deep neural networks? *NeurIPS 2014*

---

### Optuna (Hyperparameter Optimization)
**Qué es:** Framework de optimización bayesiana de hiperparámetros.

**Cuándo se usó:**
- Desde `resnet18-model-v4` (commit con "Optuna")
- Implementado en [optuna.py](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/optuna.py)

**Referencia sugerida:**
> Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. *KDD 2019*

---

### AdamW Optimizer
**Qué es:** Variante de Adam con weight decay correctamente implementado.

**Cuándo se usó:** [train_utils.py L41](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/train_utils.py#L41)

**Referencia sugerida:**
> Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*

---

### ReduceLROnPlateau (Learning Rate Scheduler)
**Qué es:** Reduce el learning rate cuando la métrica deja de mejorar.

**Cuándo se usó:** [train_utils.py L42-48](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/train_utils.py#L42-48)

**Referencia sugerida:**
(Técnica estándar, incluida en PyTorch documentation)

---

### Early Stopping
**Qué es:** Detención temprana del entrenamiento cuando no hay mejora.

**Cuándo se usó:**
- Implementación propia en [train_utils.py L5-33](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/train_utils.py#L5-33)
- Ajustado en commit `69e9ac7` (removido delta)

**Referencia sugerida:**
> Prechelt, L. (1998). Early Stopping — But When? *Neural Networks: Tricks of the Trade*

---

## 6. Funciones de Pérdida

### Cross-Entropy Loss
**Qué es:** Función de pérdida estándar para clasificación multiclase.

**Cuándo se usó:** [train_utils.py L49](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/train_utils.py#L49) - `nn.CrossEntropyLoss()`

**Referencia sugerida:**
(Técnica estándar, referencia a libro de Deep Learning)
> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

### Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
**Qué es:** Pérdida para clasificación multi-label de componentes.

**Cuándo se usó:** [train_utils.py L50](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/train_utils.py#L50)

(Misma referencia que Cross-Entropy)

---

## 7. Técnicas de Evaluación

### Top-K Accuracy
**Qué es:** Métrica que considera correcta una predicción si la clase real está entre las K más probables.

**Cuándo se usó:**
- Métricas `top_5_test` y `top_5_casia` en [training_log.csv](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/training_log.csv)
- Calculado en [evaluation.py L99-117](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/evaluation.py#L99-117)

**Referencia sugerida:**
(Técnica estándar en clasificación de imágenes, introducida en ImageNet)
> Russakovsky, O., et al. (2015). ImageNet Large Scale Visual Recognition Challenge. *IJCV*

---

### Monte Carlo Dropout
**Qué es:** Usar dropout durante inferencia para estimar incertidumbre.

**Cuándo se usó:**
- Commit `8ad3c76` (MobileNetV3+FastViT)
- Implementado en [evaluation.py L10-14](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/evaluation.py#L10-14) y L81-93

**Referencia sugerida:**
> Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. *ICML 2016*

---

## 8. Frameworks y Bibliotecas

### PyTorch
**Cuándo se usó:** Todo el proyecto

**Referencia sugerida:**
> Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019*

---

### timm (PyTorch Image Models)
**Qué es:** Biblioteca con implementaciones de arquitecturas de visión.

**Cuándo se usó:** [models.py L4](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/models.py#L4), L24

**Referencia sugerida:**
> Wightman, R. (2019). PyTorch Image Models. https://github.com/rwightman/pytorch-image-models

---

### OpenCV (cv2)
**Cuándo se usó:** Operaciones morfológicas y binarización

**Referencia sugerida:**
> Bradski, G. (2000). The OpenCV Library. *Dr. Dobb's Journal*

---

## 9. Conceptos Adicionales

### Batch Normalization
**Cuándo se usó:** Incluido en todas las arquitecturas CNN usadas

**Referencia sugerida:**
> Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training. *ICML 2015*

---

### Dropout
**Cuándo se usó:** En las arquitecturas CNN (implícito)

**Referencia sugerida:**
> Srivastava, N., et al. (2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting. *JMLR*

---

### Softmax Function
**Cuándo se usó:** Para obtener probabilidades en clasificación ([evaluation.py L87, L97](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/TRAIN/modules/evaluation.py#L87-97))

(Referencia al libro de Deep Learning de Goodfellow)

---

## Resumen por Categoría

| Categoría | Nº Refs |
|-----------|---------|
| Datasets | 3 |
| Arquitecturas CNN/ViT | 7 |
| Data Augmentation | 4 |
| Preprocesamiento | 1 |
| Optimización | 5 |
| Funciones de pérdida | 1 |
| Evaluación | 3 |
| Frameworks | 3 |
| Conceptos adicionales | 3 |
| **TOTAL** | **~30 referencias** |

---

## Referencias ya en el repositorio

Ya tienes un paper en la carpeta `REFERENCES/`:
- [High_performance_offline_handwritten_Chinese_character_recognition_using_GoogLeNet_and_directional_feature_maps.pdf](file:///home/tycho/Escritorio/TFG-MODELO-DEVELOPMENT-BACKUPSSD/REFERENCES/High_performance_offline_handwritten_Chinese_character_recognition_using_GoogLeNet_and_directional_feature_maps.pdf)

Este paper es relevante para el **estado del arte** en reconocimiento de caracteres CJK.