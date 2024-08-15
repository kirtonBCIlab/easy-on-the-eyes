# featureExtractorMSI.py
"""
Implementation of MSI feature extractor.
Zhang, Yangsong, et al. "The extension of multivariate 
synchronization index method for SSVEP-based BCI." Neurocomputing
269 (2017): 226-231
"""
# Import the definition of the parent class.  Make sure the file is in the
# working directory. 
from .featureExtractorTemplateMatching \
    import FeatureExtractorTemplateMatching

# Needed for many matrix computations.
import numpy as np

try:
    import cupy as cp
    cupy_available_global = True
except:
    cupy_available_global = False
    cp = np

class FeatureExtractorMSI(FeatureExtractorTemplateMatching):
    """Class of MSI feature extractor"""
    
    def __init__(self):        
         """MSI feature extractor class constructor"""         
         super().__init__()
                 
         # This is the covariance matrix of the template SSVEP. 
         # We can pre-compute this once to improve performance.
         self.C22 = 0
         
    def setup_feature_extractor(
            self, 
            harmonics_count,
            targets_frequencies,
            sampling_frequency,
            embedding_dimension=0,
            delay_step=0,
            filter_order=0,
            filter_cutoff_low=0,
            filter_cutoff_high=0,
            subbands=None,
            voters_count=1,
            random_seed=0,
            use_gpu=False,
            max_batch_size=16,
            explicit_multithreading=0,
            samples_count=0):
        """
        Setup the feature extractor parameters (MSI).
        
        Mandatory Parameters:
        ---------------------
        harmonics_count: The number of harmonics to be used in constructing
        the template signal.  This variable must be a positive integer 
        number (typically a value from 3 to 5).  
        
        targets_frequencies: The stimulation freqeuency of each target.  
        This must be a 1D array,  where the first element is the stimulation
        frequency of the first target, the second element is the stimulation
        frequency of the second target, and so on.  The length of this array
        indicates the number of targets.  The user must determine the 
        targets_frequencies but the number of targets (targets_count) is 
        extracted automatically from the length of targets_frequencies. 
        
        sampling_frequency: The sampling rate of the signal 
        (in samples per second).  It must be a real positive value.
        
        Optional Parameters:
        --------------------        
        embedding_dimension: This is the dimension of time-delay embedding. 
        This must be a non-negative integer.  If set to zero, no time-dely
        embedding will be used.  If there are E electrodes and we set the 
        embedding_dimension to n, the class expands the input signal as if we
        had n*E channels.  The additional channels are generated by shift_left
        operator. The number of samples that we shift each signal is 
        controlled by delay_step.  Embedding delays truncates the signal. 
        Make sure the signal is long enough. 
        
        delay_step: The number of samples that are shifted for each delay
        embedding dimension.  For example, assume we have ten channels, 
        embedding_dimension is two, and delay_step is three.  In this case, the
        class creates 30 channels.  The first ten channels are the original
        signals coming from the ten electrodes.  The second ten signals are
        obtained by shifting the origianl signals by three samples.  The third
        ten signals are obtained by shifting the original signals by six 
        samples.  The signals are truncated accordingly. 
        
        filter_order: The order of the filter used for filtering signals before
        analysis.  If filter_order is zero (the default value), no filtering
        is performed.  Otherwise, the class creates a filter of order 
        filter_order.  This must be positive integer. 
        
        cutoff_frequency_low: The first cutoff frequency of the bandpass 
        filter.  This must be a single real positive number.  If filter_order
        is zero, this attribute is ignored.  
        
        cutoff_frequency_high: The second cutoff frequency of the bandpass
        filter. This must be a single real positive number.  If filter_order
        is zero, this attribute is ignored.  
        
        subbands: This is the primary way to instruct the classifier whether 
        to use filterbank or not.  The default value is None.  If set to None, 
        the classifier uses none-fitlerbank implementation.  To use
        filterbanks, subbands must be set to a 2D array, whith exactly two 
        columns.  Each row of this matrix defines a subband with two 
        frequencies provided in two columns.  The first column is the first
        cutoff frequency and the second column is the second cutoff frequency
        of that subband.  Filterbank filters the signal using a bandpass
        filter with these cutoff frequencies to obtain a new subband.  The
        number of rows in the matrix defines the number of subbands. All
        frequencies must be in Hz.  For each row, the second column must
        always be greater than the first column. 
        
        voters_count: The number of electrode-selections that are used for
        classification.  This must be a positive integer.  This is the 
        same as the number of voters.  If voters_count is larger that the 
        cardinality of the power set of the current selected electrodes, 
        then at least one combination is bound to happen more than once. 
        However, because the selection is random, even if voters_count is
        less than the cardinality of the power set, repettitions are still
        possible (although unlikely). If not specified or 1, no 
        voting will be used. 
        
        random_seed: This parameter control the seed for random selection 
        of electrodes.  This must be set to a non-negative integer.  The 
        default value is zero.
                
        use_gpu: When set to 'True,' the class uses a gpu to extract features.
        The host must be equipped with a CUDA-capable GPU.  When set to
        'False,' all processing will be on CPU. 
        
        max_batch_size: The maximum number of signals/channel selections
        that are processed in one batch.  Increasing this number improves
        parallelization at the expense of more memory requirement.  
        This must be a single positve integer. 
        
        explicit_multithreading: This parameter determines whether to use 
        explicit multithreading or not.  If set to a non-positive integer, 
        no multithreading will be used.  If set to a positive integer, the 
        class creates multiple threads to process signals/voters in paralle.
        The number of threads is the same as the value of this variable. 
        E.g., if set to 4, the class distributes the workload among four 
        threads.  Typically, this parameter should be the same as the number
        of cores the cput has, if multithreading is to be used. 
        Multithreading cannot be used when use_gpu is set to True.
        If multithreading is set to a positive value while used_gpu is 
        set to True or vice versa, the classes raises an error and the 
        program terminates. 
        
        samples_count: If provided, the class performs precomputations that
        only depend on the number of samples, e.g., computing the template
        signal.  If not provided, the class does not perform precomputations.
        Instead, it does the computations once the input signal was provided 
        and the class learns the number of samples from the input signal. 
        Setting samples_count is highly recommended.  If the feaure extraction
        method is being used in loop (e.g., BCI2000 loop), setting this 
        parameter eliminates the need to compute the template matrix each
        time. It also helps the class to avoid other computations in each
        iteration. samples_count passed to this function must be the same 
        as the third dimension size of the signal passed to extract_features().
        If that is not the case, the template and input signal will have 
        different dimensions.  The class should issue an error in this case
        and terminate the execution. 
        """
        self.build_feature_extractor(
            harmonics_count,
            targets_frequencies,
            sampling_frequency,
            subbands=subbands,           
            embedding_dimension=embedding_dimension,
            delay_step=delay_step,
            filter_order=filter_order,
            filter_cutoff_low=filter_cutoff_low,
            filter_cutoff_high=filter_cutoff_high,
            voters_count=voters_count,
            random_seed=random_seed,
            use_gpu=use_gpu,
            max_batch_size=max_batch_size,
            explicit_multithreading=explicit_multithreading,
            samples_count=samples_count)
                        
    def get_features(self, device):
        """Extract MSI features (synchronization indexes) from signal"""                          
        signal = self.get_current_data_batch()        
        xp = self.get_array_module(signal)
        
        features = self.compute_synchronization_index(signal, device)
        batch_size = self.channel_selection_info_bundle[1]
        
        features = xp.reshape(features, (
            features.shape[0]//batch_size,
            batch_size,
            self.targets_count,
            self.features_count)
            )
        return features
    
    def get_features_multithreaded(self, signal):
        """Extract MSI features from a single signal"""        
        # Make sure signal is 3D
        signal -= np.mean(signal, axis=-1)[:, None]
        signal = signal[None, :, :] 
        features = self.compute_synchronization_index(signal, device=0)  

        # De-bundle the results.
        features = np.reshape(features, (
            1, 
            1,
            1,
            self.targets_count,
            self.features_count)
            )
          
        return features
    
    def compute_synchronization_index(self, signal, device):
        """Compute the synchronization index between signal and templates"""        
        xp = self.get_array_module(signal)        
        electrodes_count = signal.shape[1]        
        r_matrix = self.compute_r(signal, device)        
        
        # Keep only the first output of the function
        eigen_values = xp.linalg.eigh(r_matrix)[0]
        eigen_values = (eigen_values / (
            2*self.harmonics_count_handle[device]+electrodes_count)
            )

        score = xp.multiply(eigen_values, xp.log(eigen_values))
        score = xp.sum(score, axis=-1)
        score = score / xp.log(r_matrix.shape[-1])
        score += 1
        return score
      
    def compute_r(self, signal, device):
        """Compute matrix C as explained in Eq. (7)"""
        xp = self.get_array_module(signal)
        C11 = self.get_data_covariance(signal, device)
        
        C11 = xp.expand_dims(C11, axis=1)
        signal = xp.expand_dims(signal, axis=1)
        
        C12 = xp.matmul(
            signal, (self.template_signal_handle[device])[None, :, :, :])
                       
        C12 = C12 / self.samples_count_handle[device]
        electrodes_count = C11.shape[-1]
        
        # Eq. (6)
        upper_left = xp.matmul(C11, C12)
        upper_left = xp.matmul(upper_left, self.C22_handle[device])
        
        lower_right = xp.matmul(
            self.C22_handle[device], xp.transpose(C12, axes=[0, 1, 3, 2]))
        
        lower_right = xp.matmul(lower_right, C11)        
        eye1 = xp.eye(electrodes_count, dtype=xp.float32)   
        
        eye1 = eye1 + xp.zeros(
            upper_left.shape[0:2] + eye1.shape, dtype=xp.float32)
        
        eye2 = xp.eye(C12.shape[-1], dtype=xp.float32)
        
        eye2 = eye2 + xp.zeros(
            upper_left.shape[0:2] + eye2.shape, dtype=xp.float32)
        
        part1 = xp.concatenate((eye1, upper_left), axis=-1)
        part2 = xp.concatenate((lower_right, eye2), axis=-1)
        r_matrix = xp.concatenate((part1, part2), axis=-2)
       
        return xp.real(r_matrix)
                
    def get_data_covariance(self, signal, device):
        """Compute the covariance of data per Eq. (3) and Eq. (6)"""
        xp = self.get_array_module(signal)
        C11 = xp.matmul(signal, xp.transpose(signal, axes=(0, 2, 1)))
        C11 = C11 / self.samples_count_handle[device]
        C11 = xp.linalg.inv(C11)
        C11 = self.matrix_square_root(C11)
        return C11
    
    def matrix_square_root(self, A):
        """Compute the square root of a matrix using ev decomposition"""
        xp = self.get_array_module(A)
        w, v = xp.linalg.eigh(A)
        D = xp.zeros(A.shape, dtype=xp.float32)        
        w = xp.sqrt(w)
        
        for i in xp.arange(w.shape[-1]):
            D[:, i, i] = w[:, i]
            
        sqrt_A = xp.matmul(xp.matmul(v, D), xp.linalg.inv(v))
        return sqrt_A
            
    def perform_voting_initialization(self, device=0):
        """Perform initialization and precomputations common to all voters"""
        # Center data
        self.all_signals -= np.mean(self.all_signals, axis=-1)[:, :, None]        
        self.all_signals_handle = self.handle_generator(self.all_signals)
                   
    def class_specific_initializations(self):
        """Perform necessary initializations"""
        # Perform some percomputations only in the first run.  
        # These computations only rely on the template signal and can thus
        # be pre-computed to improve performance. 
        self.compute_templates()  
        self.precompute_template_covariance()
        
        # Create handles
        # Handles make it easier to expand the algorithm to work with
        # multiple CPUs or GPUs
        self.template_signal_handle = self.handle_generator(
            self.template_signal) 
        self.C22_handle = self.handle_generator(self.C22)
        
        self.harmonics_count_handle = self.handle_generator(
            self.harmonics_count)
        
        # It is important to cast these to float32.
        # Otherwise, cupy casts the results back to float64 when dividing
        # float32 by int32.
        self.samples_count_handle = self.handle_generator(
            np.float32(self.samples_count))
        
    def precompute_template_covariance(self):
        """Pre-compute and save the covariance matrix of the template"""
        # Eq. (4)
        C22 = np.matmul(            
            np.transpose(self.template_signal, axes=[0, 2, 1]),
            self.template_signal)
        
        C22 = C22 / self.samples_count
        
        # Eq. (6)
        C22 = np.linalg.inv(C22)            
        self.C22 = self.matrix_square_root(C22)      
               
    def get_current_data_batch(self):
        """Bundle all data so they can be processed toegher"""
        # Bundling helps increase GPU and CPU utilization. 
       
        # Extract bundle information. 
        # Helps with the code's readability. 
        batch_index = self.channel_selection_info_bundle[0]        
        batch_population = self.channel_selection_info_bundle[1]
        batch_electrodes_count = self.channel_selection_info_bundle[2]
        first_signal = self.channel_selection_info_bundle[3]
        last_signal = self.channel_selection_info_bundle[4]
        signals_count = last_signal - first_signal
        
        # Pre-allocate memory for the batch
        signal = np.zeros(
            (signals_count, batch_population,
             batch_electrodes_count, self.samples_count),
            dtype=np.float32)        
        
        selected_signals = self.all_signals_handle[0][first_signal:last_signal]
        
        for j in np.arange(batch_population):
            current_selection = self.channel_selections[batch_index]
            signal[:, j] = selected_signals[:, current_selection, :]
            batch_index += 1
                        
        signal = np.reshape(signal, (-1,) + signal.shape[2:])
          
        # Move the extracted batches to the device memory if need be. 
        if self.use_gpu == True:          
            signal = cp.asarray(signal)
            
        return signal