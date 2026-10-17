"""
Open-Set Dataset Splitting and Protocol Management.
Divides anomaly categories into Known (available during training) and
Unseen/Held-out (reserved strictly for open-set evaluation).
"""
from typing import Dict, List, Optional, Set, Tuple
import random


class OpenSetSplitManager:
    """
    Manages known vs. unseen anomaly category partitions and splits.
    """
    DEFAULT_BENCHMARK_CATEGORIES = {
        "ucf_crime": [
            "Abuse", "Arrest", "Arson", "Assault", "Burglary",
            "Explosion", "Fighting", "RoadAccidents", "Robbery",
            "Shooting", "Shoplifting", "Stealing", "Vandalism"
        ],
        "shanghaitech": [
            "BikingOnFootpath", "Chasing", "Fighting", "Jumping",
            "Skateboarding", "VehicleAnomaly", "Littering"
        ]
    }

    def __init__(
        self,
        benchmark_name: str = "ucf_crime",
        custom_known: Optional[List[str]] = None,
        custom_unseen: Optional[List[str]] = None,
        unseen_ratio: float = 0.35,
        seed: int = 42
    ):
        self.benchmark_name = benchmark_name
        self.seed = seed
        random.seed(seed)

        if custom_known is not None and custom_unseen is not None:
            self.known_categories = list(custom_known)
            self.unseen_categories = list(custom_unseen)
        else:
            all_cats = self.DEFAULT_BENCHMARK_CATEGORIES.get(
                benchmark_name.lower(),
                self.DEFAULT_BENCHMARK_CATEGORIES["ucf_crime"]
            )
            shuffled = list(all_cats)
            random.shuffle(shuffled)
            num_unseen = max(1, int(len(shuffled) * unseen_ratio))
            self.unseen_categories = sorted(shuffled[:num_unseen])
            self.known_categories = sorted(shuffled[num_unseen:])

        # Always include 'Normal' as known base class
        if "Normal" not in self.known_categories:
            self.known_categories.insert(0, "Normal")

    def get_summary(self) -> Dict[str, object]:
        """Returns overview of the open-set split protocol."""
        return {
            "benchmark": self.benchmark_name,
            "num_known": len(self.known_categories),
            "known_categories": self.known_categories,
            "num_unseen": len(self.unseen_categories),
            "unseen_categories": self.unseen_categories,
            "openness": 1.0 - (len(self.known_categories) / (len(self.known_categories) + len(self.unseen_categories)))
        }

    def is_category_unseen(self, category_name: str) -> bool:
        """Checks if a given anomaly category is part of the held-out unseen set."""
        return category_name in self.unseen_categories

    def map_category_to_label(self, category_name: str) -> Tuple[int, str]:
        """
        Maps category name to integer index and open-set status string:
        - ('Normal', 0, 'Normal')
        - ('Fighting', idx, 'Known Anomaly')
        - ('Arson', -1, 'Unseen Anomaly')
        """
        if category_name == "Normal":
            return 0, "Normal"
        if category_name in self.known_categories:
            return self.known_categories.index(category_name), "Known Anomaly"
        if category_name in self.unseen_categories:
            return -1, "Unseen Anomaly"
        return -1, "Novel Anomaly"
