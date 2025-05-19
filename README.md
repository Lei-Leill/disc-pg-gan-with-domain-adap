# Progress Writeup 

## CNN with GRL Layer Testing
- Tested and verified functionality of CNN with GRL layer to ensure it meets performance expectations.
- Ensured network structure consistency between versions with and without GRL.
- Used `vscode-remote://ssh-remote%2Bwest.cs.haverford.edu/homes/tlei/mathiesonlab/disc-pg-gan-with-domain_adap/figs/CHB/CEU/Class_Train_Test_Accuracy_vs_Epoch.pdf` for structural comparison.

## Sequence Setup
- Correctly configured discriminator and classifier labels in a dictionary format for neural network branches.
- Discussed and clarified domain adaptation structure.

## With/Without Comparison with Simulated Human Data
- Implemented and compared CNN performance with and without GRL using SlimIterator with simulated data.
- Conducted tests (`GRL_test_sim.py`) training on one population (CEU, CHB, YRI) and testing on another.
- Visualized results including Accuracy per selection group, Classifier Train/Test Accuracy, Discriminator Train/Test Accuracy, and Confusion Matrix. Results saved in the `figs` folder.
- Found no significant difference between models with and without GRL.

## With/Without Comparison with Human vs. Mosquito Data 
- Tested CNN performance using human simulated data (CEU) and tested against simulated mosquito data.
- Expected significant domain differences between populations.
- Used `GRL_test_mosq.py` for testing.
- Analyzed figures in the `figs/Mosq` folder.

### Observations:
- **With GRL:** Noticed decreasing testing loss trend and stable testing accuracy range around 12-15 epochs, indicating better learning and performance.

- **Without GRL:** Testing loss fluctuated without a clear decreasing trend. Training accuracy remained high but did not fit well to the testing mosquito dataset.

### Next Steps:
- Clean up the code to reduce redundancy and simplify `if/else` statements for better abstraction and generalization.
