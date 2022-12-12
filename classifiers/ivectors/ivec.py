# # Generated with SMOP  0.41
# from libsmop import *
# # ivec.m
#
#     ## Speaker Verification Using i-Vectors
# # Speaker verification, or authentication, is the task of confirming that the
# # identity of a speaker is who they purport to be. Speaker verification has been
# # an active research area for many years. An early performance breakthrough was
# # to use a Gaussian mixture model and universal background model (GMM-UBM) [1]
# # on acoustic features (usually <docid:audio_ref#mw_b965edcf-ff08-417b-992c-b7f6471536c8
# # fe_spafe>). For an example, see <docid:audio_ug#mw_e25a4f73-e645-469c-bb0a-3b5b809e1cdf
# # Speaker Verification Using Gaussian Mixture Models>. One of the main difficulties
# # of GMM-UBM systems involves intersession variability. Joint factor analysis
# # (JFA) was proposed to compensate for this variability by separately modeling
# # inter-speaker variability and channel or session variability [2] [3]. However,
# # [4] discovered that channel factors in the JFA also contained information about
# # the speakers, and proposed combining the channel and speaker spaces into a _total
# # variability_ _space_. Intersession variability was then compensated for by using
# # backend procedures, such as linear discriminant analysis (LDA) and within-class
# # covariance normalization (WCCN), followed by a scoring, such as the cosine similarity
# # score. [5] proposed replacing the cosine similarity scoring with a probabilistic
# # LDA (PLDA) model. [11] and [12] proposed a method to Gaussianize the i-vectors
# # and therefore make Gaussian assumptions in the PLDA, referred to as G-PLDA or
# # simplified PLDA. While i-vectors were originally proposed for speaker verification,
# # they have been applied to many problems, like language recognition, speaker
# # diarization, emotion recognition, age estimation, and anti-spoofing [10]. Recently,
# # deep learning techniques have been proposed to replace i-vectors with _d-vectors_
# # or _x-vectors_ [8] [6].
# ## Use an i-Vector System
# # Audio Toolbox provides <docid:audio_ref#mw_16f16947-a396-480e-9953-1f91c81885f0
# # |ivectorSystem|> which encapsulates the ability to train an i-vector system,
# # enroll speakers or other audio labels, evaluate the system for a decision threshold,
# # and identify or verify speakers or other audio labels. See <docid:audio_ref#mw_16f16947-a396-480e-9953-1f91c81885f0
# # |ivectorSystem|> for examples of using this feature and applying it to several
# # applications.
# #
# # To learn more about how an i-vector system works, continue with the example.
# ## Develop an i-Vector System
# # In this example, you develop a standard i-vector system for speaker verification
# # that uses an LDA-WCCN backend with either cosine similarity scoring or a G-PLDA
# # scoring.
# #
# #
# #
# # Throughout the example, you will find live controls on tunable parameters.
# # Changing the controls does not rerun the example. If you change a control, you
# # must rerun the example.
# ## Data Set Management
# # This example uses the Pitch Tracking Database from Graz University of Technology
# # (PTDB-TUG) [7]. The data set consists of 20 English native speakers reading
# # 2342 phonetically rich sentences from the TIMIT corpus. Download and extract
# # the data set. Depending on your system, downloading and extracting the data
# # set can take approximately 1.5 hours.
#
#     # url = "https://www2.spsc.tugraz.at/databases/PTDB-TUG/SPEECH_DATA_ZIPPED.zip";
#     downloadFolder=copy(tempdir)
#     datasetFolder='/Users/david/Documents/data/speech/ivectors'
#     # if ~datasetExists(datasetFolder)
# #     disp("Downloading PTDB-TUG (3.9 G) ...")
# #     unzip(url,datasetFolder)
# # end
# ##
# # Create an <docid:audio_ref#mw_6315b106-9a7b-4a11-a7c6-322c073e343a |audioDatastore|>
# # object that points to the data set. The data set was originally intended for
# # use in pitch-tracking training and evaluation, and includes laryngograph readings
# # and baseline pitch decisions. Use only the original audio recordings.
#
#     ads=audioDatastore(concat([fullfile(datasetFolder,'SPEECH DATA','FEMALE','MIC'),fullfile(datasetFolder,'SPEECH DATA','MALE','MIC')]),IncludeSubfolders=copy(true),FileExtensions='.wav')
#     fileNames=ads.Files
#     ##
# # The file names contain the speaker IDs. Decode the file names to set the labels
# # on the |audioDatastore| object.
#
#     speakerIDs=extractBetween(fileNames,'mic_','_')
#     ads.Labels = copy(categorical(speakerIDs))
#     countEachLabel(ads)
#     ##
# # Separate the |audioDatastore| object into training, evaluation, and test sets.
# # The training set contains 16 speakers. The evaluation set contains four speakers
# # and is further divided into an enrollment set and a set to evaluate the detection
# # error tradeoff of the trained i-vector system, and a test set.
#
#     developmentLabels=categorical(concat(['M01','M02','M03','M04','M06','M07','M08','M09','F01','F02','F03','F04','F06','F07','F08','F09']))
#     evaluationLabels=categorical(concat(['M05','M10','F05','F10']))
#     adsTrain=subset(ads,ismember(ads.Labels,developmentLabels))
#     adsEvaluate=subset(ads,ismember(ads.Labels,evaluationLabels))
#     numFilesPerSpeakerForEnrollment=3
#     adsEnroll,adsTest,adsDET=splitEachLabel(adsEvaluate,numFilesPerSpeakerForEnrollment,2,nargout=3)
#     ##
# # Display the label distributions of the resulting |audioDatastore| objects.
#
#     countEachLabel(adsTrain)
#     countEachLabel(adsEnroll)
#     countEachLabel(adsDET)
#     countEachLabel(adsTest)
#     ##
# # Read an audio file from the training data set, listen to it, and plot it.
# # Reset the datastore.
#
#     audio,audioInfo=read(adsTrain,nargout=2)
#     fs=audioInfo.SampleRate
#     t=(arange(0,size(audio,1) - 1)) / fs
#     sound(audio,fs)
#     plot(t,audio)
#     xlabel('Time (s)')
#     ylabel('Amplitude')
#     axis(concat([0,t(end()),- 1,1]))
#     title('Sample Utterance from Training Set')
#     reset(adsTrain)
#     ##
# # You can reduce the data set and the number of parameters used in this example
# # to speed up the runtime at the cost of performance. In general, reducing the
# # data set is a good practice for development and debugging.
#
#     speedUpExample=copy(true)
#     if speedUpExample:
#         adsTrain=splitEachLabel(adsTrain,10)
#         adsDET=splitEachLabel(adsDET,10)
#
#     ## Feature Extraction
# # Create an <docid:audio_ref#mw_b56cd7dc-af31-4da4-a43e-b13debc30322 |audioFeatureExtractor|>
# # object to extract 20 MFCCs, 20 delta-MFCCs, and 20 delta-delta MFCCs. Use a
# # delta window length of 9. Extract features from 25 ms Hann windows with a 10
# # ms hop.
#
#     numCoeffs=20
#     deltaWindowLength=9
#     windowDuration=0.025
#     hopDuration=0.01
#     windowSamples=round(dot(windowDuration,fs))
#     hopSamples=round(dot(hopDuration,fs))
#     overlapSamples=windowSamples - hopSamples
#     afe=audioFeatureExtractor(SampleRate=copy(fs),Window=hann(windowSamples,'periodic'),OverlapLength=copy(overlapSamples),fe_spafe=copy(true),mfccDelta=copy(true),mfccDeltaDelta=copy(true))
#     setExtractorParameters(afe,'fe_spafe',DeltaWindowLength=copy(deltaWindowLength),NumCoeffs=copy(numCoeffs))
#     ##
# # Extract features from the audio read from the training datastore. Features
# # are returned as a |numHops|-by-|numFeatures| matrix.
#
#     features=extract(afe,audio)
#     numHops,numFeatures=size(features,nargout=2)
#     ## Training
# # Training an i-vector system is computationally expensive and time-consuming.
# # If you have Parallel Computing Toolbox™, you can spread the work across multiple
# # cores to speed up the example. Determine the optimal number of partitions for
# # your system. If you do not have Parallel Computing Toolbox™, use a single partition.
#
#     if logical_not(isempty(ver('parallel'))) and logical_not(speedUpExample):
#         pool=copy(gcp)
#         numPar=numpartitions(adsTrain,pool)
#     else:
#         numPar=1
#
#     ## Feature Normalization Factors
# # Use the helper function, |helperFeatureExtraction|, to extract all features
# # from the data set. The |helperFeatureExtraction| function extracts MFCC from
# # regions of speech in the audio. The speech detection is performed by the <docid:audio_ref#mw_f7b40697-af02-4c71-a508-ecd8f7f47400
# # |detectSpeech|> function.
#
#     featuresAll=cellarray([])
#     tic
#     for ii in arange(1,numPar).reshape(-1):
#         adsPart=partition(adsTrain,numPar,ii)
#         featuresPart=cell(0,numel(adsPart.Files))
#         for iii in arange(1,numel(adsPart.Files)).reshape(-1):
#             audioData=read(adsPart)
#             featuresPart[iii]=helperFeatureExtraction(audioData,afe,[])
#         featuresAll=concat([featuresAll,featuresPart])
#
#     allFeatures=cat(2,featuresAll[arange()])
#     disp('Feature extraction from training set complete (' + toc + ' seconds).')
#     ##
# # Calculate the global mean and standard deviation of each feature. You will
# # use these in future calls to the |helperFeatureExtraction| function to normalize
# # the features.
#
#     normFactors.Mean = copy(mean(allFeatures,2,'omitnan'))
#     normFactors.STD = copy(std(allFeatures,[],2,'omitnan'))
#     ## Universal Background Model (UBM)
# # Initialize the Gaussian mixture model (GMM) that will be the universal background
# # model (UBM) in the i-vector system. The component weights are initialized as
# # evenly distributed. Systems trained on the TIMIT data set usually contain around
# # 2048 components.
#
#     numComponents=64
#     if speedUpExample:
#         numComponents=32
#
#     alpha=ones(1,numComponents) / numComponents
#     mu=randn(numFeatures,numComponents)
#     vari=rand(numFeatures,numComponents) + eps
#     ubm=struct(ComponentProportion=copy(alpha),mu=copy(mu),sigma=copy(vari))
#     ##
# # Train the UBM using the expectation-maximization (EM) algorithm.
#
#     maxIter=10
#     if speedUpExample:
#         maxIter=2
#
#     tic
#     for iter in arange(1,maxIter).reshape(-1):
#         tic
#         # EXPECTATION
#         N=zeros(1,numComponents)
#         F=zeros(numFeatures,numComponents)
#         S=zeros(numFeatures,numComponents)
#         L=0
#         for ii in arange(1,numPar).reshape(-1):
#             adsPart=partition(adsTrain,numPar,ii)
#             while hasdata(adsPart):
#
#                 audioData=read(adsPart)
#                 Y=helperFeatureExtraction(audioData,afe,normFactors)
#                 logLikelihood=helperGMMLogLikelihood(Y,ubm)
#                 amax=max(logLikelihood,[],1)
#                 logLikelihoodSum=amax + log(sum(exp(logLikelihood - amax),1))
#                 gamma=exp(logLikelihood - logLikelihoodSum).T
#                 n=sum(gamma,1)
#                 f=dot(Y,gamma)
#                 s=dot((multiply(Y,Y)),gamma)
#                 N=N + n
#                 F=F + f
#                 S=S + s
#                 L=L + sum(logLikelihoodSum)
#
#         # Print current log-likelihood
#         disp('Training UBM: ' + iter + '/' + maxIter + ' complete (' + round(toc) + ' seconds), Log-likelihood = ' + round(L))
#         # MAXIMIZATION
#         N=max(N,eps)
#         ubm.ComponentProportion = copy(max(N / sum(N),eps))
#         ubm.ComponentProportion = copy(ubm.ComponentProportion / sum(ubm.ComponentProportion))
#         ubm.mu = copy(F / N)
#         ubm.sigma = copy(max(S / N - ubm.mu ** 2,eps))
#
#     ## Calculate Baum-Welch Statistics
# # The Baum-Welch statistics are the |_N_| (zeroth order) and |_F_| (first order)
# # statistics used in the EM algorithm, calculated using the final UBM.
# #
# # $$N_c \left(s\right)=\sum_t \gamma_t \left(c\right)$$
# #
# # $$F_c \left(s\right)=\sum_t \gamma_t \left(c\right)Y_t$$
# ##
# # * $Y_t$ is the feature vector at time $t$.
# # * $s\in \left\lbrace s_1 ,s_{2,} \ldotp \ldotp \ldotp ,s_N \right\rbrace$,
# # where $N$ is the number of speakers. For the purposes of training the total
# # variability space, each audio file is considered a separate speaker (whether
# # or not it belongs to a physical single speaker).
# # * $\gamma_t \left(c\right)$ is the posterior probability that the UBM component
# # $c$ accounts for the feature vector $Y_t$.
# ##
# # Calculate the zeroth and first order Baum-Welch statistics over the training
# # set.
# # change this to just a fe extractor then can work out if BW stats are
# # correct...
#
#     numSpeakers=numel(adsTrain.Files)
#     Nc=cellarray([])
#     Fc=cellarray([])
#     numFiles=numel(adsTrain.Files)
#     all_features=cell(0,numel(adsPart.Files))
#     tic
#     for ii in arange(1,numPar).reshape(-1):
#         adsPart=partition(adsTrain,numPar,ii)
#         numFiles=numel(adsPart.Files)
#         Npart=cell(1,numFiles)
#         Fpart=cell(1,numFiles)
#         for jj in arange(1,numFiles).reshape(-1):
#             audioData=read(adsPart)
#             Y=helperFeatureExtraction(audioData,afe,normFactors)
#             all_features[jj]=Y
#             logLikelihood=helperGMMLogLikelihood(Y,ubm)
#             amax=max(logLikelihood,[],1)
#             logLikelihoodSum=amax + log(sum(exp(logLikelihood - amax),1))
#             gamma=exp(logLikelihood - logLikelihoodSum).T
#             n=sum(gamma,1)
#             f=dot(Y,gamma)
#             Npart[jj]=reshape(n,1,1,numComponents)
#             Fpart[jj]=reshape(f,numFeatures,1,numComponents)
#         Nc=concat([Nc,Npart])
#         Fc=concat([Fc,Fpart])
#
#     disp('Baum-Welch statistics completed (' + toc + ' seconds).')
#     ##
# # ComponentProportion = ubm.ComponentProportion;
# # mu = ubm.mu;
# # sigma = ubm.sigma;
# # save('ComponentProportion.mat', 'ComponentProportion')
# # save('sigma.mat', 'sigma')
# # save('mu.mat', 'mu')
#
#     ##
# # Expand the statistics into matrices and center $F\left(s\right)$, as described
# # in [3], such that
# ##
# # * $N\left(s\right)$ is a $C\;F\times C\;F$ diagonal matrix whose blocks are
# # $N_c \left(s\right)I\;\left(c=1,\ldotp \ldotp \ldotp C\right)$.
# # * $F\left(s\right)$ is a $C\;F\times 1$ supervector obtained by concatenating
# # $F_c \left(s\right)\;\;\left(c=1,\ldotp \ldotp \ldotp C\right)$.
# # * $C$ is the number of components in the UBM.
# # * $F$ is the number of features in a feature vector.
#
#     N=copy(Nc)
#     F=copy(Fc)
#     muc=reshape(ubm.mu,numFeatures,1,[])
#     for s in arange(1,numSpeakers).reshape(-1):
#         N[s]=repelem(reshape(Nc[s],1,[]),numFeatures)
#         F[s]=reshape(Fc[s] - multiply(Nc[s],muc),[],1)
#
#     ##
#
#     # test = Fc{s};
# # size(test)
# # test(1,1,2)
#     save('F.mat','F')
#     ##
# # Because this example assumes a diagonal covariance matrix for the UBM, |N|
# # are also diagonal matrices, and are saved as vectors for efficient computation.
# ## Total Variability Space
# # In the i-vector model, the ideal speaker supervector consists of a speaker-independent
# # component and a speaker-dependent component. The speaker-dependent component
# # consists of the total variability space model and the speaker's i-vector.
# #
# # $$M=m+\textrm{Tw}$$
# ##
# # * $M$ is the speaker utterance supervector
# # * $m$ is the speaker- and channel-independent supervector, which can be taken
# # to be the UBM supervector.
# # * $T$ is a low-rank rectangular matrix and represents the total variability
# # subspace.
# # * $w$ is the i-vector for the speaker
# ##
# # The dimensionality of the i-vector, $w$, is typically much lower than the
# # C F -dimensional speaker utterance supervector, making the i-vector, or i-vectors,
# # a much more compact and tractable representation.
# #
# # To train the total variability space, $T$, first randomly initialize |T|,
# # then perform these steps iteratively [3]:
# ##
# # # Calculate the posterior distribution of the hidden variable.
# ##
# # $$l_T \left(s\right)=I+T^{\prime } \times \Sigma^{-1} \times N\left(s\right)\times
# # T$$
# #
# # 2. Accumulate statistics across the speakers.
# #
# # $$K=\sum_s F\left(s\right)\times {\left(l_T^{-1} \left(s\right)\times T^{\prime
# # } \times \Sigma^{-1} \times F\left(s\right)\right)}^{\prime }$$
# #
# # $$A_c =\sum_s N_c \left(s\right)l_T^{-1} \left(s\right)$$
# #
# # 3. Update the total variability space.
# #
# # $$T_c =A_c^{-1} \times K$$
# #
# # $$T=\left\lbrack \begin{array}{c}T_1 \\T_2 \\\vdots \\T_C \end{array}\right\rbrack$$
# #
# # [3] proposes initializing $\Sigma$ by the UBM variance, and then updating
# # $\Sigma$ according to the equation:
# #
# # $$\Sigma ={\left(\sum_s N\left(s\right)\right)}^{-1} \left(\left(\sum_s S\left(s\right)\right)-\textrm{diag}\left(K\times
# # T^{\prime } \right)\right)$$
# #
# # where S(s) is the centered second-order Baum-Welch statistic. However, updating
# # $\Sigma$ is often dropped in practice as it has little effect. This example
# # does not update $\Sigma$.
# ##
# # Create the sigma variable.
#
#     Sigma=ravel(ubm.sigma)
#     ##
# # Specify the dimension of the total variability space. A typical value used
# # for the TIMIT data set is 1000.
#
#     numTdim=32
#     if speedUpExample:
#         numTdim=16
#
#     ##
# # Initialize |T| and the identity matrix, and preallocate cell arrays.
#
#     T=randn(numel(ubm.sigma),numTdim)
#     T=T / norm(T)
#     I=eye(numTdim)
#     Ey=cell(numSpeakers,1)
#     Eyy=cell(numSpeakers,1)
#     Linv=cell(numSpeakers,1)
#     ##
# # Set the number of iterations for training. A typical value reported is 20.
#     save('T.mat','T')
#     numIterations=5
#     ##
# # Run the training loop.
# # vars to cpoy over
#
#     for iterIdx in arange(1,numIterations).reshape(-1):
#         tic
#         # 1. Calculate the posterior distribution of the hidden variable
#         TtimesInverseSSdiag=(T / Sigma).T
#         for s in arange(1,numSpeakers).reshape(-1):
#             L=(I + dot(multiply(TtimesInverseSSdiag,N[s]),T))
#             Linv[s]=pinv(L)
#             Ey[s]=dot(dot(Linv[s],TtimesInverseSSdiag),F[s])
#             Eyy[s]=Linv[s] + dot(Ey[s],Ey[s].T)
#         # 2. Accumlate statistics across the speakers
#         Eymat=cat(2,Ey[arange()])
#         FFmat=cat(2,F[arange()])
#         Kt=dot(FFmat,Eymat.T)
#         K=mat2cell(Kt.T,numTdim,repelem(numFeatures,numComponents))
#         newT=cell(numComponents,1)
#         for c in arange(1,numComponents).reshape(-1):
#             AcLocal=zeros(numTdim)
#             for s in arange(1,numSpeakers).reshape(-1):
#                 AcLocal=AcLocal + dot(Nc[s](arange(),arange(),c),Eyy[s])
#             # 3. Update the Total Variability Space
#             newT[c]=(dot(pinv(AcLocal),K[c])).T
#         T=cat(1,newT[arange()])
#         disp('Training Total Variability Space: ' + iterIdx + '/' + numIterations + ' complete (' + round(toc,2) + ' seconds).')
#
#     ## i-Vector Extraction
# # Once the total variability space is calculated, you can calculate the i-vectors
# # as [4]:
# #
# # $$w={\left(I+T^{\prime } \Sigma^{-1\;} \textrm{NT}\right)}^{\prime } T^{\prime
# # } \Sigma^{-1\;} F$$
# #
# # At this point, you are still considering each training file as a separate
# # speaker. However, in the next step, when you train a projection matrix to reduce
# # dimensionality and increase inter-speaker differences, the i-vectors must be
# # labeled with the appropriate, distinct speaker IDs.
# #
# # Create a cell array where each element of the cell array contains a matrix
# # of i-vectors across files for a particular speaker.
#
#     speakers=unique(adsTrain.Labels)
#     numSpeakers=numel(speakers)
#     ivectorPerSpeaker=cell(numSpeakers,1)
#     TS=T / Sigma
#     TSi=TS.T
#     ubmMu=ubm.mu
#     tic
#     for speakerIdx in arange(1,numSpeakers).reshape(-1):
#         # Subset the datastore to the speaker you are adapting.
#         adsPart=subset(adsTrain,adsTrain.Labels == speakers(speakerIdx))
#         numFiles=numel(adsPart.Files)
#         ivectorPerFile=zeros(numTdim,numFiles)
#         for fileIdx in arange(1,numFiles).reshape(-1):
#             audioData=read(adsPart)
#             Y=helperFeatureExtraction(audioData,afe,normFactors)
#             logLikelihood=helperGMMLogLikelihood(Y,ubm)
#             amax=max(logLikelihood,[],1)
#             logLikelihoodSum=amax + log(sum(exp(logLikelihood - amax),1))
#             gamma=exp(logLikelihood - logLikelihoodSum).T
#             n=sum(gamma,1)
#             f=dot(Y,gamma) - multiply(n,(ubmMu))
#             ivectorPerFile[arange(),fileIdx]=dot(dot(pinv(I + dot((multiply(TS,repelem(ravel(n),numFeatures))).T,T)),TSi),ravel(f))
#         ivectorPerSpeaker[speakerIdx]=ivectorPerFile
#
#     disp('I-vectors extracted from training set (' + toc + ' seconds).')
#     ## Projection Matrix
# # Many different backends have been proposed for i-vectors. The most straightforward
# # and still well-performing one is the combination of linear discriminant analysis
# # (LDA) and within-class covariance normalization (WCCN).
# #
# # Create a matrix of the training vectors and a map indicating which i-vector
# # corresponds to which speaker. Initialize the projection matrix as an identity
# # matrix.
#
#     w=copy(ivectorPerSpeaker)
#     utterancePerSpeaker=cellfun(lambda x=None: size(x,2),w)
#     ivectorsTrain=cat(2,w[arange()])
#     projectionMatrix=eye(size(w[1],1))
#     ##
# # LDA attempts to minimize the intra-class variance and maximize the variance
# # between speakers. It can be calculated as outlined in [4]:
# #
# # *Given*:
# #
# # $$S_b =\sum_{s=1}^S \left(\bar{w_s } -\bar{w} \right){\left(\bar{w_s } -\bar{w}
# # \right)}^{\prime }$$
# #
# # $$S_w =\sum_{s=1}^S \frac{1}{n_s }\sum_{i=1}^{n_s } \left(w_i^s -\bar{w_s
# # } \right){\left(w_i^s -\bar{w_s } \right)}^{\prime }$$
# #
# # where
# ##
# # * $\bar{w_s } =\left(\frac{1}{n_s }\right)\sum_{i=1}^{n_s } w_i^s$ is the
# # mean of i-vectors for each speaker.
# # * $\bar{w} =\frac{1}{N}\sum_{s=1}^S \sum_{i=1}^{n_s } w_i^s$ is the mean i-vector
# # across all speakers.
# # * $n_s$ is the number of utterances for each speaker.
# ##
# # *Solve* the eigenvalue equation for the best eigenvectors:
# #
# # $$S_b v=\lambda \;S_w v$$
# #
# # The best eigenvectors are those with the highest eigenvalues.
#
#     performLDA=copy(true)
#     if performLDA:
#         tic
#         numEigenvectors=16
#         Sw=zeros(size(projectionMatrix,1))
#         Sb=zeros(size(projectionMatrix,1))
#         wbar=mean(cat(2,w[arange()]),2)
#         for ii in arange(1,numel(w)).reshape(-1):
#             ws=w[ii]
#             wsbar=mean(ws,2)
#             Sb=Sb + dot((wsbar - wbar),(wsbar - wbar).T)
#             Sw=Sw + cov(ws.T,1)
#         A,__=eigs(Sb,Sw,numEigenvectors,nargout=2)
#         A=(A / vecnorm(A)).T
#         ivectorsTrain=dot(A,ivectorsTrain)
#         w=mat2cell(ivectorsTrain,size(ivectorsTrain,1),utterancePerSpeaker)
#         projectionMatrix=dot(A,projectionMatrix)
#         disp('LDA projection matrix calculated (' + round(toc,2) + ' seconds).')
#
#     ##
# # WCCN attempts to scale the i-vector space inversely to the in-class covariance,
# # so that directions of high intra-speaker variability are de-emphasized in i-vector
# # comparisons [9].
# #
# # *Given* the within-class covariance matrix:
# #
# # $$W=\frac{1}{S}\sum_{s=1}^S \frac{1}{n_s }\sum_{i=1}^{n_s } \left(w_i^s -\bar{w_s
# # } \right){\left(w_i^s -\bar{w_s } \right)}^{\prime }$$
# #
# # where
# ##
# # * $\bar{w_s } =\left(\frac{1}{n_s }\right)\sum_{i=1}^{n_s } w_i^s$ is the
# # mean of i-vectors for each speaker.
# # * $n_s$ is the number of utterances for each speaker.
# ##
# # *Solve* for B using Cholesky decomposition:
# #
# # $$W^{-1} ={\textrm{BB}}^{\prime }$$
#
#     performWCCN=copy(true)
#     if performWCCN:
#         tic
#         alpha=0.9
#         W=zeros(size(projectionMatrix,1))
#         for ii in arange(1,numel(w)).reshape(-1):
#             W=W + cov(w[ii].T,1)
#         W=W / numel(w)
#         W=dot((1 - alpha),W) + dot(alpha,eye(size(W,1)))
#         B=chol(pinv(W),'lower')
#         projectionMatrix=dot(B,projectionMatrix)
#         disp('WCCN projection matrix calculated (' + round(toc,4) + ' seconds).')
#
#     ##
# # The training stage is now complete. You can now use the universal background
# # model (UBM), total variability space (T), and projection matrix to enroll and
# # verify speakers.
# ## Train G-PLDA Model
# # Apply the projection matrix to the train set.
#
#     ivectors=cellfun(lambda x=None: dot(projectionMatrix,x),ivectorPerSpeaker,UniformOutput=copy(false))
#     ##
# # This algorithm implemented in this example is a Gaussian PLDA as outlined
# # in [13]. In the Gaussian PLDA, the i-vector is represented with the following
# # equation:
# #
# # $$\phi_{\textrm{ij}} =\mu +{\textrm{Vy}}_i +\varepsilon_{\textrm{ij}}$$
# #
# # $$y_i \sim N\left(0,I\right)$$
# #
# # $$\varepsilon_{\textrm{ij}} \sim N\left(0,\Lambda^{-1} \right)$$
# #
# # where $\mu \;$is a global mean of the i-vectors, $\Lambda$ is a full precision
# # matrix of the noise term $\varepsilon_{\textrm{ij}}$, and $V$ is the factor
# # loading matrix, also known as the eigenvoices.
# #
# # Specify the number of eigenvoices to use. Typically numbers are between 10
# # and 400.
#
#     numEigenVoices=16
#     ##
# # Determine the number of disjoint persons, the number of dimensions in the
# # feature vectors, and the number of utterances per speaker.
#
#     K=numel(ivectors)
#     D=size(ivectors[1],1)
#     utterancePerSpeaker=cellfun(lambda x=None: size(x,2),ivectors)
#     ##
# # Find the total number of samples and center the i-vectors.
# #
# # $$N=\sum_{i=1}^K n_i$$
# #
# # $$\mu =\frac{1}{N}\sum_{i,j} \phi_{i,j}$$
# #
# # $$\varphi_{\textrm{ij}} =\phi_{\textrm{ij}} -\mu$$
#
#     ivectorsMatrix=cat(2,ivectors[arange()])
#     N=size(ivectorsMatrix,2)
#     mu=mean(ivectorsMatrix,2)
#     ivectorsMatrix=ivectorsMatrix - mu
#     ##
# # Determine a whitening matrix from the training i-vectors and then whiten the
# # i-vectors. Specify either ZCA whitening, PCA whitening, or no whitening.
#
#     whiteningType='ZCA'
#     if strcmpi(whiteningType,'ZCA'):
#         S=cov(ivectorsMatrix.T)
#         __,sD,sV=svd(S,nargout=3)
#         W=dot(diag(1.0 / (sqrt(diag(sD)) + eps)),sV.T)
#         ivectorsMatrix=dot(W,ivectorsMatrix)
#     else:
#         if strcmpi(whiteningType,'PCA'):
#             S=cov(ivectorsMatrix.T)
#             sV,sD=eig(S,nargout=2)
#             W=dot(diag(1.0 / (sqrt(diag(sD)) + eps)),sV.T)
#             ivectorsMatrix=dot(W,ivectorsMatrix)
#         else:
#             W=eye(size(ivectorsMatrix,1))
#
#     ##
# # Apply length normalization and then convert the training i-vector matrix back
# # to a cell array.
#
#     ivectorsMatrix=ivectorsMatrix / vecnorm(ivectorsMatrix)
#     ##
# # Compute the global second-order moment as
# #
# # $$S=\sum_{\textrm{ij}} \varphi_{\textrm{ij}} \varphi_{\textrm{ij}}^T$$
#
#     S=dot(ivectorsMatrix,ivectorsMatrix.T)
#     ##
# # Convert the training i-vector matrix back to a cell array.
#
#     ivectors=mat2cell(ivectorsMatrix,D,utterancePerSpeaker)
#     ##
# # Sort persons according to the number of samples and then group the i-vectors
# # by number of utterances per speaker. Precalculate the first-order moment of
# # the $i$-th person as
# #
# # $$f_i =\sum_{j=1}^{n_i } \varphi_{\textrm{ij}}$$
#
#     uniqueLengths=unique(utterancePerSpeaker)
#     numUniqueLengths=numel(uniqueLengths)
#     speakerIdx=1
#     f=zeros(D,K)
#     for uniqueLengthIdx in arange(1,numUniqueLengths).reshape(-1):
#         idx=find(utterancePerSpeaker == uniqueLengths(uniqueLengthIdx))
#         temp=cellarray([])
#         for speakerIdxWithinUniqueLength in arange(1,numel(idx)).reshape(-1):
#             rho=ivectors(idx(speakerIdxWithinUniqueLength))
#             temp=concat([[temp],[rho]])
#             f[arange(),speakerIdx]=sum(rho[arange()],2)
#             speakerIdx=speakerIdx + 1
#         ivectorsSorted[uniqueLengthIdx]=temp
#
#     ##
# # Initialize the eigenvoices matrix, V, and the inverse noise variance term,
# # $\Lambda$.
#
#     V=randn(D,numEigenVoices)
#     Lambda=pinv(S / N)
#     ##
# # Specify the number of iterations for the EM algorithm and whether or not to
# # apply the minimum divergence.
#
#     numIter=5
#     minimumDivergence=copy(true)
#     ##
# # Train the G-PLDA model using the EM algorithm described in [13].
#
#     for iter in arange(1,numIter).reshape(-1):
#         # EXPECTATION
#         gamma=zeros(numEigenVoices,numEigenVoices)
#         EyTotal=zeros(numEigenVoices,K)
#         R=zeros(numEigenVoices,numEigenVoices)
#         idx=1
#         for lengthIndex in arange(1,numUniqueLengths).reshape(-1):
#             ivectorLength=uniqueLengths(lengthIndex)
#             iv=ivectorsSorted[lengthIndex]
#             M=pinv(dot(ivectorLength,(dot(V.T,(dot(Lambda,V))))) + eye(numEigenVoices))
#             # Loop over each speaker for the current i-vector length
#             for speakerIndex in arange(1,numel(iv)).reshape(-1):
#                 # First moment of latent variable for V
#                 Ey=dot(dot(dot(M,V.T),Lambda),f(arange(),idx))
#                 # Calculate second moment.
#                 Eyy=dot(Ey,Ey.T)
#                 R=R + dot(ivectorLength,(M + Eyy))
#                 # Append EyTotal
#                 EyTotal[arange(),idx]=Ey
#                 idx=idx + 1
#                 if minimumDivergence:
#                     gamma=gamma + (M + Eyy)
#         # Calculate T
#         TT=dot(EyTotal,f.T)
#         # MAXIMIZATION
#         V=dot(TT.T,pinv(R))
#         Lambda=pinv((S - dot(V,TT)) / N)
#         # MINIMUM DIVERGENCE
#         if minimumDivergence:
#             gamma=gamma / K
#             V=dot(V,chol(gamma,'lower'))
#
#     ##
# # Once you've trained the G-PLDA model, you can use it to calculate a score
# # based on the log-likelihood ratio as described in [14]. Given two i-vectors
# # that have been centered, whitened, and length-normalized, the score is calculated
# # as:
# #
# # $$\textrm{score}\left(w_1 ,w_t \right)=\left\lbrack \begin{array}{cc}w_1^T
# # & w_t^T \end{array}\right\rbrack \left\lbrack \begin{array}{cc}\Sigma +{\textrm{VV}}^T
# # & {\textrm{VV}}^T \\{\textrm{VV}}^T  & \Sigma +{\textrm{VV}}^T \end{array}\right\rbrack
# # \left\lbrack \begin{array}{cc}w_1  & w_t \end{array}\right\rbrack -w_1^T {\left\lbrack
# # \begin{array}{c}\Sigma +{\textrm{VV}}^T \end{array}\right\rbrack }^{-1} w_1
# # -w_t^T {\left\lbrack \begin{array}{c}\Sigma +{\textrm{VV}}^T \end{array}\right\rbrack
# # }^{-1} w_t +C$$
# #
# # where $w_1$ and $w_t$ are the enrollment and test i-vectors, $\Sigma$ is the
# # variance matrix of the noise term, $\mathrm{V}$ is the eigenvoice matrix. The
# # $C$ term are factored-out constants and can be dropped in practice.
#
#     speakerIdx=2
#     utteranceIdx=1
#     w1=ivectors[speakerIdx](arange(),utteranceIdx)
#     speakerIdx=1
#     utteranceIdx=10
#     wt=ivectors[speakerIdx](arange(),utteranceIdx)
#     VVt=dot(V,V.T)
#     SigmaPlusVVt=pinv(Lambda) + VVt
#     term1=pinv(concat([[SigmaPlusVVt,VVt],[VVt,SigmaPlusVVt]]))
#     term2=pinv(SigmaPlusVVt)
#     w1wt=concat([[w1],[wt]])
#     score=dot(dot(w1wt.T,term1),w1wt) - dot(dot(w1.T,term2),w1) - dot(dot(wt.T,term2),wt)
#     ##
# # In practice, the test i-vectors, and depending on your system, the enrollment
# # ivectors, are not used in the training of the G-PLDA model. In the following
# # evaluation section, you use previously unseen data for enrollment and verification.
# # The supporting function, gpldaScore encapsulates the scoring steps above, and
# # additionally performs centering, whitening, and normalization. Save the trained
# # G-PLDA model as a struct for use with the supporting function |gpldaScore|.
#
#     gpldaModel=struct(mu=copy(mu),WhiteningMatrix=copy(W),EigenVoices=copy(V),Sigma=pinv(Lambda))
#     ## Enroll
# # Enroll new speakers that were not in the training data set.
# #
# # Create i-vectors for each file for each speaker in the enroll set using the
# # this sequence of steps:
# ##
# # # Feature Extraction
# # # Baum-Welch Statistics: Determine the zeroth and first order statistics
# # # i-vector Extraction
# # # Intersession compensation
# ##
# # Then average the i-vectors across files to create an i-vector model for the
# # speaker. Repeat the for each speaker.
#
#     speakers=unique(adsEnroll.Labels)
#     numSpeakers=numel(speakers)
#     enrolledSpeakersByIdx=cell(numSpeakers,1)
#     tic
#     for speakerIdx in arange(1,numSpeakers).reshape(-1):
#         # Subset the datastore to the speaker you are adapting.
#         adsPart=subset(adsEnroll,adsEnroll.Labels == speakers(speakerIdx))
#         numFiles=numel(adsPart.Files)
#         ivectorMat=zeros(size(projectionMatrix,1),numFiles)
#         for fileIdx in arange(1,numFiles).reshape(-1):
#             audioData=read(adsPart)
#             Y=helperFeatureExtraction(audioData,afe,normFactors)
#             logLikelihood=helperGMMLogLikelihood(Y,ubm)
#             amax=max(logLikelihood,[],1)
#             logLikelihoodSum=amax + log(sum(exp(logLikelihood - amax),1))
#             gamma=exp(logLikelihood - logLikelihoodSum).T
#             n=sum(gamma,1)
#             f=dot(Y,gamma) - multiply(n,(ubmMu))
#             w=dot(dot(pinv(I + dot((multiply(TS,repelem(ravel(n),numFeatures))).T,T)),TSi),ravel(f))
#             w=dot(projectionMatrix,w)
#             ivectorMat[arange(),fileIdx]=w
#         # i-vector model
#         enrolledSpeakersByIdx[speakerIdx]=mean(ivectorMat,2)
#
#     disp('Speakers enrolled (' + round(toc,2) + ' seconds).')
#     ##
# # For bookkeeping purposes, convert the cell array of i-vectors to a structure,
# # with the speaker IDs as fields and the i-vectors as values
#
#     enrolledSpeakers=copy(struct)
#     for s in arange(1,numSpeakers).reshape(-1):
#         setattr(enrolledSpeakers,string(speakers(s)),enrolledSpeakersByIdx[s])
#
#     ## Verification
# # Specify either the CSS or G-PLDA scoring method.
#
#     scoringMethod='CSS'
#     ## False Rejection Rate (FRR)
# # The speaker false rejection rate (FRR) is the rate that a given speaker is
# # incorrectly rejected. Create an array of scores for enrolled speaker i-vectors
# # and i-vectors of the same speaker.
#
#     speakersToTest=unique(adsDET.Labels)
#     numSpeakers=numel(speakersToTest)
#     scoreFRR=cell(numSpeakers,1)
#     tic
#     for speakerIdx in arange(1,numSpeakers).reshape(-1):
#         adsPart=subset(adsDET,adsDET.Labels == speakersToTest(speakerIdx))
#         numFiles=numel(adsPart.Files)
#         ivectorToTest=getattr(enrolledSpeakers,(string(speakersToTest(speakerIdx))))
#         score=zeros(numFiles,1)
#         for fileIdx in arange(1,numFiles).reshape(-1):
#             audioData=read(adsPart)
#             Y=helperFeatureExtraction(audioData,afe,normFactors)
#             logLikelihood=helperGMMLogLikelihood(Y,ubm)
#             amax=max(logLikelihood,[],1)
#             logLikelihoodSum=amax + log(sum(exp(logLikelihood - amax),1))
#             gamma=exp(logLikelihood - logLikelihoodSum).T
#             n=sum(gamma,1)
#             f=dot(Y,gamma) - multiply(n,(ubmMu))
#             w=dot(dot(pinv(I + dot((multiply(TS,repelem(ravel(n),numFeatures))).T,T)),TSi),ravel(f))
#             w=dot(projectionMatrix,w)
#             if strcmpi(scoringMethod,'CSS'):
#                 score[fileIdx]=dot(ivectorToTest,w) / (dot(norm(w),norm(ivectorToTest)))
#             else:
#                 score[fileIdx]=gpldaScore(gpldaModel,w,ivectorToTest)
#         scoreFRR[speakerIdx]=score
#
#     disp('FRR calculated (' + round(toc,2) + ' seconds).')
#     ## False Acceptance Rate (FAR)
# # The speaker false acceptance rate (FAR) is the rate that utterances not belonging
# # to an enrolled speaker are incorrectly accepted as belonging to the enrolled
# # speaker. Create an array of scores for enrolled speakers and i-vectors of different
# # speakers.
#
#     speakersToTest=unique(adsDET.Labels)
#     numSpeakers=numel(speakersToTest)
#     scoreFAR=cell(numSpeakers,1)
#     tic
#     for speakerIdx in arange(1,numSpeakers).reshape(-1):
#         adsPart=subset(adsDET,adsDET.Labels != speakersToTest(speakerIdx))
#         numFiles=numel(adsPart.Files)
#         ivectorToTest=getattr(enrolledSpeakers,(string(speakersToTest(speakerIdx))))
#         score=zeros(numFiles,1)
#         for fileIdx in arange(1,numFiles).reshape(-1):
#             audioData=read(adsPart)
#             Y=helperFeatureExtraction(audioData,afe,normFactors)
#             logLikelihood=helperGMMLogLikelihood(Y,ubm)
#             amax=max(logLikelihood,[],1)
#             logLikelihoodSum=amax + log(sum(exp(logLikelihood - amax),1))
#             gamma=exp(logLikelihood - logLikelihoodSum).T
#             n=sum(gamma,1)
#             f=dot(Y,gamma) - multiply(n,(ubmMu))
#             w=dot(dot(pinv(I + dot((multiply(TS,repelem(ravel(n),numFeatures))).T,T)),TSi),ravel(f))
#             w=dot(projectionMatrix,w)
#             if strcmpi(scoringMethod,'CSS'):
#                 score[fileIdx]=dot(ivectorToTest,w) / (dot(norm(w),norm(ivectorToTest)))
#             else:
#                 score[fileIdx]=gpldaScore(gpldaModel,w,ivectorToTest)
#         scoreFAR[speakerIdx]=score
#
#     disp('FAR calculated (' + round(toc,2) + ' seconds).')
#     ## Equal Error Rate (EER)
# # To compare multiple systems, you need a single metric that combines the FAR
# # and FRR performance. For this, you determine the equal error rate (EER), which
# # is the threshold where the FAR and FRR curves meet. In practice, the EER threshold
# # might not be the best choice. For example, if speaker verification is used as
# # part of a multi-authentication approach for wire transfers, FAR would most likely
# # be more heavily weighted than FRR.
#
#     amin=min(cat(1,scoreFRR[arange()],scoreFAR[arange()]))
#     amax=max(cat(1,scoreFRR[arange()],scoreFAR[arange()]))
#     thresholdsToTest=linspace(amin,amax,1000)
#     # Compute the FRR and FAR for each of the thresholds.
#     if strcmpi(scoringMethod,'CSS'):
#         # In CSS, a larger score indicates the enroll and test ivectors are
#     # similar.
#         FRR=mean(cat(1,scoreFRR[arange()]) < thresholdsToTest)
#         FAR=mean(cat(1,scoreFAR[arange()]) > thresholdsToTest)
#     else:
#         # In G-PLDA, a smaller score indicates the enroll and test ivectors are
#     # similar.
#         FRR=mean(cat(1,scoreFRR[arange()]) > thresholdsToTest)
#         FAR=mean(cat(1,scoreFAR[arange()]) < thresholdsToTest)
#
#     __,EERThresholdIdx=min(abs(FAR - FRR),nargout=2)
#     EERThreshold=thresholdsToTest(EERThresholdIdx)
#     EER=mean(concat([FAR(EERThresholdIdx),FRR(EERThresholdIdx)]))
#     figure
#     plot(thresholdsToTest,FAR,'k',thresholdsToTest,FRR,'b',EERThreshold,EER,'ro',MarkerFaceColor='r')
#     title(concat(['Equal Error Rate = ' + round(EER,4),'Threshold = ' + round(EERThreshold,4)]))
#     xlabel('Threshold')
#     ylabel('Error Rate')
#     legend('False Acceptance Rate (FAR)','False Rejection Rate (FRR)','Equal Error Rate (EER)',Location='best')
#     grid('on')
#     axis('tight')
#     ## Supporting Functions
# ## Feature Extraction and Normalization
#
#
# @function
# def helperFeatureExtraction(audioData=None,afe=None,normFactors=None,*args,**kwargs):
#     varargin = helperFeatureExtraction.varargin
#     nargin = helperFeatureExtraction.nargin
#
#     # Input:
#     # audioData   - column vector of audio data
#     # afe         - audioFeatureExtractor object
#     # normFactors - mean and standard deviation of the features used for normalization.
#     #               If normFactors is empty, no normalization is applied.
#
#     # Output
#     # features    - matrix of features extracted
#     # numFrames   - number of frames (feature vectors) returned
#
#     # Normalize
#     audioData=audioData / max(abs(ravel(audioData)))
#
#     audioData[isnan(audioData)]=0
#
#     idx=detectSpeech(audioData,afe.SampleRate)
#     features=[]
#     for ii in arange(1,size(idx,1)).reshape(-1):
#         f=extract(afe,audioData(arange(idx(ii,1),idx(ii,2))))
#         features=concat([[features],[f]])
#
#     # Feature normalization
#     if logical_not(isempty(normFactors)):
#         features=(features - normFactors.Mean.T) / normFactors.STD.T
#
#     features=features.T
#
#     if logical_not(isempty(normFactors)):
#         features=features - mean(features,'all')
#
#
#     numFrames=size(features,2)
#     return features,numFrames
#
# if __name__ == '__main__':
#     pass
#
#     ## Gaussian Multi-Component Mixture Log-Likelihood
#
#
# @function
# def helperGMMLogLikelihood(x=None,gmm=None,*args,**kwargs):
#     varargin = helperGMMLogLikelihood.varargin
#     nargin = helperGMMLogLikelihood.nargin
#
#     xMinusMu=repmat(x,1,1,numel(gmm.ComponentProportion)) - permute(gmm.mu,concat([1,3,2]))
#     permuteSigma=permute(gmm.sigma,concat([1,3,2]))
#     Lunweighted=dot(- 0.5,(sum(log(permuteSigma),1) + sum(multiply(xMinusMu,(xMinusMu / permuteSigma)),1) + dot(size(gmm.mu,1),log(dot(2,pi)))))
#     temp=squeeze(permute(Lunweighted,concat([1,3,2])))
#     if size(temp,1) == 1:
#         # If there is only one frame, the trailing singleton dimension was
#         # removed in the permute. This accounts for that edge case.
#         temp=temp.T
#
#
#     L=temp + log(gmm.ComponentProportion).T
#     return L
#
# if __name__ == '__main__':
#     pass
#
#     ## G-PLDA Score
#
#
# @function
# def gpldaScore(gpldaModel=None,w1=None,wt=None,*args,**kwargs):
#     varargin = gpldaScore.varargin
#     nargin = gpldaScore.nargin
#
#     # Center the data
#     w1=w1 - gpldaModel.mu
#     wt=wt - gpldaModel.mu
#     # Whiten the data
#     w1=dot(gpldaModel.WhiteningMatrix,w1)
#     wt=dot(gpldaModel.WhiteningMatrix,wt)
#     # Length-normalize the data
#     w1=w1 / vecnorm(w1)
#     wt=wt / vecnorm(wt)
#     # Score the similarity of the i-vectors based on the log-likelihood.
#     VVt=dot(gpldaModel.EigenVoices,gpldaModel.EigenVoices.T)
#     SVVt=gpldaModel.Sigma + VVt
#     term1=pinv(concat([[SVVt,VVt],[VVt,SVVt]]))
#     term2=pinv(SVVt)
#     w1wt=concat([[w1],[wt]])
#     score=dot(dot(w1wt.T,term1),w1wt) - dot(dot(w1.T,term2),w1) - dot(dot(wt.T,term2),wt)
#     return score
#
# if __name__ == '__main__':
#     pass
#
#     ## References
# # [1] Reynolds, Douglas A., et al. "Speaker Verification Using Adapted Gaussian
# # Mixture Models." _Digital Signal Processing_, vol. 10, no. 1–3, Jan. 2000, pp.
# # 19–41. _DOI.org (Crossref)_, doi:10.1006/dspr.1999.0361.
# #
# # [2] Kenny, Patrick, et al. "Joint Factor Analysis Versus Eigenchannels in
# # Speaker Recognition." _IEEE Transactions on Audio, Speech and Language Processing_,
# # vol. 15, no. 4, May 2007, pp. 1435–47. _DOI.org (Crossref)_, doi:10.1109/TASL.2006.881693.
# #
# # [3] Kenny, P., et al. "A Study of Interspeaker Variability in Speaker Verification."
# # _IEEE Transactions on Audio, Speech, and Language Processing_, vol. 16, no.
# # 5, July 2008, pp. 980–88. _DOI.org (Crossref)_, doi:10.1109/TASL.2008.925147.
# #
# # [4] Dehak, Najim, et al. "Front-End Factor Analysis for Speaker Verification."
# # _IEEE Transactions on Audio, Speech, and Language Processing_, vol. 19, no.
# # 4, May 2011, pp. 788–98. _DOI.org (Crossref)_, doi:10.1109/TASL.2010.2064307.
# #
# # [5] Matejka, Pavel, Ondrej Glembek, Fabio Castaldo, M.j. Alam, Oldrich Plchot,
# # Patrick Kenny, Lukas Burget, and Jan Cernocky. "Full-Covariance UBM and Heavy-Tailed
# # PLDA in i-Vector Speaker Verification." _2011 IEEE International Conference
# # on Acoustics, Speech and Signal Processing (ICASSP)_, 2011. https://doi.org/10.1109/icassp.2011.5947436.
# #
# # [6] Snyder, David, et al. "X-Vectors: Robust DNN Embeddings for Speaker Recognition."
# # _2018 IEEE International Conference on Acoustics, Speech and Signal Processing
# # (ICASSP)_, IEEE, 2018, pp. 5329–33. _DOI.org (Crossref)_, doi:10.1109/ICASSP.2018.8461375.
# #
# # [7] Signal Processing and Speech Communication Laboratory. Accessed December
# # 12, 2019. <https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html.
# # https://www.spsc.tugraz.at/databases-and-tools/ptdb-tug-pitch-tracking-database-from-graz-university-of-technology.html.>
# #
# # [8] Variani, Ehsan, et al. "Deep Neural Networks for Small Footprint Text-Dependent
# # Speaker Verification." _2014 IEEE International Conference on Acoustics, Speech
# # and Signal Processing (ICASSP)_, IEEE, 2014, pp. 4052–56. _DOI.org (Crossref)_,
# # doi:10.1109/ICASSP.2014.6854363.
# #
# # [9] Dehak, Najim, Réda Dehak, James R. Glass, Douglas A. Reynolds and Patrick
# # Kenny. “Cosine Similarity Scoring without Score Normalization Techniques.” _Odyssey_
# # (2010).
# #
# # [10] Verma, Pulkit, and Pradip K. Das. “I-Vectors in Speech Processing Applications:
# # A Survey.” _International Journal of Speech Technology_, vol. 18, no. 4, Dec.
# # 2015, pp. 529–46. _DOI.org (Crossref)_, doi:10.1007/s10772-015-9295-3.
# #
# # [11] D. Garcia-Romero and C. Espy-Wilson, “Analysis of I-vector Length Normalization
# # in Speaker Recognition Systems.” _Interspeech_, 2011, pp. 249–252.
# #
# # [12] Kenny, Patrick. "Bayesian Speaker Verification with Heavy-Tailed Priors".
# # _Odyssey 2010 - The Speaker and Language Recognition Workshop_, Brno, Czech
# # Republic, 2010.
# #
# # [13] Sizov, Aleksandr, Kong Aik Lee, and Tomi Kinnunen. “Unifying Probabilistic
# # Linear Discriminant Analysis Variants in Biometric Authentication.” _Lecture
# # Notes in Computer Science Structural, Syntactic, and Statistical Pattern Recognition_,
# # 2014, 464–75. https://doi.org/10.1007/978-3-662-44415-3_47.
# #
# # [14] Rajan, Padmanabhan, Anton Afanasyev, Ville Hautamäki, and Tomi Kinnunen.
# # 2014. “From Single to Multiple Enrollment I-Vectors: Practical PLDA Scoring
# # Variants for Speaker Verification.” _Digital Signal Processing_ 31 (August):
# # 93–101. https://doi.org/10.1016/j.dsp.2014.05.001.
# #
# # _Copyright 2019 The MathWorks, Inc._