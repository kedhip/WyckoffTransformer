class StatisticalEvaluator():
    def __init__(self,
                 test_dataset: pd.DataFrame,
                 train_dataset: Optional[pd.DataFrame] = None):
        self.test_dataset = test_dataset
        self.dof_counter = None
        if train_dataset is not None:
            self.train_fingerprints = frozenset(chain.from_iterable(train_dataset.apply(record_to_augmented_fingerprints, axis=1)))
            with open(Path(__file__).parent.parent.resolve() / "cache" / "wychoffs_enumerated_by_ss.pkl.gz", "rb") as f:
                self.letter_to_enum, _ , self.letter_to_ss = pickle.load(f)[:3]
            self.generated_to_fingerprint = partial(
                generated_to_fingerprint, letter_to_ss=self.letter_to_ss, letter_to_enum=self.letter_to_enum)
        self.test_novelty = None
        self.test_sg_counts = self.test_dataset['spacegroup_number'].value_counts()

    def get_test_novelty(self):
        # Just a single augmentation variant per structure
        if self.test_novelty is None:
            test_fp = self.test_dataset.apply(record_to_augmented_fingerprints, axis=1).map(lambda x: next(iter(x)))
            self.test_novelty = sum((fp not in self.train_fingerprints for fp in test_fp)) / len(test_fp)
        return self.test_novelty

    def get_num_sites_ks(self, generated_structures: Iterable[Dict], return_counts:bool=False) -> float:
        """
        Computes the Kolmogorov-Smirnov statistic between the numbers of sites in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """
        test_num_sites = self.test_dataset['site_symmetries'].map(len)
        generated_num_sites = [sum(map(len, wygene['sites'])) for wygene in generated_structures]
        if return_counts:
            return kstest(test_num_sites, generated_num_sites), generated_num_sites
        else:
            return kstest(test_num_sites, generated_num_sites)


    def get_num_elements_ks(self, generated_structures: Iterable[Dict], return_counts:bool=False) -> float:
        """
        Computes the Kolmogorov-Smirnov statistic between the numbers of elements in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """
        test_num_elements = self.test_dataset['elements'].map(count_unique)
        generated_num_elements = [len(frozenset(record["species"])) for record in generated_structures]
        if return_counts:
            return kstest(test_num_elements, generated_num_elements), generated_num_elements
        else:
            return kstest(test_num_elements, generated_num_elements)

    def get_dof_ks(self, generated_structures: Iterable[Dict]) -> float:
        """
        Computes the Kolmogorov-Smirnov statistic between the degrees of freedom in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """
        if self.dof_counter is None:
            self.dof_counter = DoFCounter()
        test_dof = self.test_dataset['dof'].apply(sum)
        generated_dof = [self.dof_counter.get_total_dof(record) for record in generated_structures]
        return kstest(test_dof, generated_dof)
    
    def get_sg_chi2(self,
        generated_structures: Iterable[Dict],
        sample_size: Optional[int|str] = None,
        return_counts: bool = False) -> float|Tuple[float, np.ndarray]:
        """
        Computes the chi-squared statistic between the space group numbers in the
        generated structures and the test dataset.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
            sample_size: If int, the number of samples to use from the generated dataset.
                if "test", use the same number as in the test dataset. NOTE: it takes first
                N samples without shuffling in the interest of reproducibility.
        """
        if sample_size is None:
            sample = generated_structures
        elif isinstance(sample_size, int):
            sample = generated_structures[:sample_size]
        elif sample_size == "test":
            sample = generated_structures[:len(self.test_dataset)]
        else:
            raise ValueError(f"Unknown sample_size {sample_size}")
        generated_sg = [record['group'] for record in sample]
        generated_sg_counts = pd.Series(generated_sg).value_counts()
        if not generated_sg_counts.index.isin(self.test_sg_counts.index).all():
            logger.warning("Generated dataset has extra space groups compared to test")
        generated_sg_counts = generated_sg_counts.reindex_like(self.test_sg_counts).fillna(0)
        chi2 = chi2_contingency(np.vstack([self.test_sg_counts, generated_sg_counts]))
        if return_counts:
            return chi2, generated_sg_counts
        else:
            return chi2


    def get_elements_chi2(self,
        generated_structures: Iterable[Dict],
        sample_size: Optional[int|str] = None,
        return_counts: bool = False) -> float|Tuple[float, np.ndarray]:
        """
        Computes the chi-squared statistic between the element counts in the
        generated structures and the test dataset.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
            sample_size: If int, the number of samples to use from the generated dataset.
                if "test", use the same number as in the test dataset. NOTE: it takes first
                N samples without shuffling in the interest of reproducibility.
        """
        if sample_size is None:
            sample = generated_structures
        elif isinstance(sample_size, int):
            sample = generated_structures[:sample_size]
        elif sample_size == "test":
            sample = generated_structures[:len(self.test_dataset)]
        else:
            raise ValueError(f"Unknown sample_size {sample_size}")
        generated_element_counts = Counter()
        for record in sample:
            for element, multiplicity in zip(record['species'], record['numIons']):
                generated_element_counts[Element(element)] += multiplicity
        test_element_counts = self.test_dataset['composition'].apply(Counter).sum()
        test_element_counts = pd.Series(test_element_counts)
        generated_counts = pd.Series(generated_element_counts)
        if not generated_counts.index.isin(test_element_counts.index).all():
            logger.warning("Generated dataset has extra elements compared to test")
        generated_counts = generated_counts.reindex_like(test_element_counts).fillna(0)
        
        chi2 = chi2_contingency(np.vstack([test_element_counts, generated_counts]))
        if return_counts:
            return chi2, generated_counts, test_element_counts
        else:
            return chi2
        

    def count_novel(self, generated_structures: Iterable[Dict]) -> float:
        generated_fp = list(map(self.generated_to_fingerprint, generated_structures))
        return sum((fp not in self.train_fingerprints for fp in generated_fp)) / len(generated_fp)

    
    def get_novel(self, generated_structures: Iterable[Dict], return_index=False) -> List[Dict]:
        generated_fp = list(map(self.generated_to_fingerprint, generated_structures))
        if return_index:
            res = [(record, i) for i, (record, fp) in enumerate(zip(generated_structures, generated_fp)) if fp not in self.train_fingerprints]
            return list(zip(*res))
        else:
            return [record for record, fp in zip(generated_structures, generated_fp) if fp not in self.train_fingerprints]
