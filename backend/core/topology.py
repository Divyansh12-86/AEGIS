"""
Minimal electrical topology graph.
Models which assets sit on which feeders, and feeder dependencies,
so Digital Engineer can check downstream/upstream impact of a strategy.
"""

from dataclasses import dataclass, field

@dataclass
class FeederNode:
    feeder_id: str
    upstream: str | None = None          # parent feeder, if any
    downstream: list[str] = field(default_factory=list)  # child feeders
    assets: list[str] = field(default_factory=list)      # asset_ids on this feeder
    critical: bool = False               # e.g. hospital/life-safety feeder


class TopologyGraph:
    def __init__(self):
        self._feeders: dict[str, FeederNode] = {}

    def add_feeder(self, feeder_id: str, upstream: str | None = None, critical: bool = False):
        node = FeederNode(feeder_id=feeder_id, upstream=upstream, critical=critical)
        self._feeders[feeder_id] = node
        if upstream and upstream in self._feeders:
            self._feeders[upstream].downstream.append(feeder_id)

    def attach_asset(self, feeder_id: str, asset_id: str):
        if feeder_id in self._feeders:
            self._feeders[feeder_id].assets.append(asset_id)

    def get_feeder(self, feeder_id: str) -> FeederNode | None:
        return self._feeders.get(feeder_id)

    def is_critical_downstream(self, feeder_id: str) -> bool:
        """Check if this feeder or anything downstream of it is critical."""
        node = self._feeders.get(feeder_id)
        if not node:
            return False
        if node.critical:
            return True
        return any(self.is_critical_downstream(child) for child in node.downstream)

    def affected_assets(self, feeder_id: str) -> list[str]:
        """All assets on this feeder and everything downstream of it."""
        node = self._feeders.get(feeder_id)
        if not node:
            return []
        assets = list(node.assets)
        for child in node.downstream:
            assets.extend(self.affected_assets(child))
        return assets


# Default in-memory topology instance (MVP — replace with DB-backed later)
topology = TopologyGraph()
topology.add_feeder("main_feeder")
topology.add_feeder("hospital_feeder", upstream="main_feeder", critical=True)
topology.add_feeder("plant_feeder_a", upstream="main_feeder")
