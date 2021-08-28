TocoDecoy: a new approach to design unbiased datasets for training and benchmarking machine-learning scoring functions

Models used in this study:
  CRNN model for decoys generation:https://github.com/pcko1/Deep-Drug-Coder
  IGN model used for MLSFs construction:https://github.com/zjujdj/InteractionGraphNet/tree/master
  XGBoost model used for MLSFs construction:https://github.com/tqchen/xgboost

Datasets used in this study:
  TocoDecoy:https://zenodo.org/record/5290011#.YSmecN--vVg
  LIT-PCBA:http://drugdesign.unistra.fr/LIT-PCBA
  
Architecture of the codes:
  1.dataset_generation:
  1.1 molecular generation: generate decoys with similar physicochemical properties based on the 'seed' active ligands using the CRNN model.
  1.2 postpreprocessing: select decoys with dissimilar topology structures to the active ligands and do the grid filter.
  2.model_training:
  2.1 IGN: scripts used for training and testing IGN
  2.2 XGB: scripts used for training and testing XGBoost
  3.utilities: scripts for various utilities
