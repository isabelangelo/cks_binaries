(1) load_training_data:
- loads the CKS flux + labels and filters them to assemble a training set.
- saves training data to the following files:
	./data/cannon_training_data/training_flux_dwt.csv : wavelet-filtered trainining flux for each object
	./data/cannon_training_data/training_sigma_dwt.csv : wavelet-filtered trainining sigma for each object

	./data/cannon_training_data/training_flux_original.csv : specmatch-emp output trainining flux for each object
	./data/cannon_training_data/training_sigma_original.csv : specmatch-emp output trainining sigma for each object