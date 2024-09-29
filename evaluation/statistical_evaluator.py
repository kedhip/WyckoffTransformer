from typing import Dict, Iterable, List, Optional, Tuple
from collections import Counter
import logging
from scipy.stats import kstest, chi2_contingency
import pandas as pd
import numpy as np
from pymatgen.core import Element

from wyckoff_transformer.evaluation import DoFCounter, count_unique

logger = logging.getLogger(__name__)

class StatisticalEvaluator():
    def __init__(self,
                 test_dataset: pd.DataFrame):

        self.test_dataset = test_dataset

        self.test_sg_counts = self.test_dataset['spacegroup_number'].value_counts()
        self.test_num_sites = self.test_dataset['site_symmetries'].map(len)
        self.test_num_elements = self.test_dataset['elements'].map(count_unique)
        self.dof_counter = DoFCounter()
        self.test_dof = self.test_dataset['dof'].apply(sum)

        test_element_counts = self.test_dataset['composition'].apply(Counter).sum()
        self.test_element_counts = pd.Series(test_element_counts)

    def get_num_sites_ks(self, generated_structures: pd.DataFrame, return_counts:bool=False):
        """
        Computes the Kolmogorov-Smirnov statistic between the numbers of sites in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        Returns:
            Kolmogorov-Smirnov object with the statistic and p-value.
        """
        generated_num_sites = generated_structures.sites.map(lambda x: sum(map(len, x)))

        if return_counts:
            return kstest(self.test_num_sites, generated_num_sites), generated_num_sites
        else:
            return kstest(self.test_num_sites, generated_num_sites)


    def get_num_elements_ks(self, generated_structures: pd.DataFrame, return_counts:bool=False) -> float:
        """
        Computes the Kolmogorov-Smirnov statistic between the numbers of elements in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """
        
        generated_num_elements = generated_structures.species.map(lambda x: len(frozenset(x)))

        if return_counts:
            return kstest(self.test_num_elements, generated_num_elements), generated_num_elements
        else:
            return kstest(self.test_num_elements, generated_num_elements)

    def get_dof_ks(self, generated_structures: pd.DataFrame) -> float:
        """
        Computes the Kolmogorov-Smirnov statistic between the degrees of freedom in the 
        generated structures and the test dataset. Note that Kolmorogov-Smirnov statistic doesn't
        a priory depend on the number examples, but p-value does.
        Args:
            generated_structures: Iterable of dictionaries with the generated structures in
                pyxtal.from_random arguments format.
        """ 
        if "dof" in generated_structures.columns:
            dofs = generated_structures.dof.apply(sum)
        else:
            dofs = generated_structures.apply(self.dof_counter.get_total_dof, axis=1)
        return kstest(self.test_dof, dofs)
    
    def get_sg_chi2(self,
        generated_structures: pd.DataFrame,
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
            sample = generated_structures.iloc[:sample_size]
        elif sample_size == "test":
            sample = generated_structures.iloc[:len(self.test_dataset)]
        else:
            raise ValueError(f"Unknown sample_size {sample_size}")
        generated_sg = sample.spacegroup_number
        generated_sg_counts = pd.Series(generated_sg).value_counts()
        if not generated_sg_counts.index.isin(self.test_sg_counts.index).all():
            logger.warning("Generated dataset has extra space groups compared to test")
        generated_sg_counts = generated_sg_counts.reindex_like(self.test_sg_counts).fillna(0)
        contingency_table = pd.concat([self.test_sg_counts, generated_sg_counts], axis=1)
        chi2 = chi2_contingency(contingency_table).statistic
        
        if return_counts:
            return chi2, generated_sg_counts
        else:
            return chi2


    def get_elements_chi2(self,
        generated_structures: pd.DataFrame,
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
        for _, record in sample.iterrows():
            for element, multiplicity in zip(record['species'], record['numIons']):
                generated_element_counts[Element(element)] += multiplicity
        
        generated_counts = pd.Series(generated_element_counts)
        if not generated_counts.index.isin(self.test_element_counts.index).all():
            logger.warning("Generated dataset has extra elements compared to test")
        generated_counts = generated_counts.reindex_like(self.test_element_counts).fillna(0)
        
        chi2 = chi2_contingency(np.vstack([self.test_element_counts, generated_counts])).statistic
        if return_counts:
            return chi2, generated_counts, self.test_element_counts
        else:
            return chi2
