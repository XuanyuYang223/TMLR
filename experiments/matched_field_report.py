"""Report the independent support-matched replication with its true bases."""
from . import field_symmetry_report, field_symmetry_transfer_report
from .matched_field import controlled_world


def run():
    field_symmetry_report.world = controlled_world
    field_symmetry_report.summarize('results/field_matched_support', verify=True)
    field_symmetry_transfer_report.summarize('results/field_matched_support_transfer')


if __name__ == '__main__': run()
