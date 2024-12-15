from enum import Enum

class ExperimentState(Enum):
    QG_DG = "QG_DG"
    QG_DAG = "QG_DAG"
    QAG_DG = "QAG_DG"
    QAG_DAG = "QAG_DAG"
    QG_QAG_DG = "QG_QAG_DG"
    QG_QAG_DAG = "QG_QAG_DAG"
    QG_DG_DAG = "QG_DG_DAG"
    QAG_DG_DAG = "QAG_DG_DAG"
    QG_QAG_DG_DAG = "QG_QAG_DG_DAG"

class ExperimentQG(Enum):
    QG_ONLY = "QG"
    QAG_ONLY = "QAG"
    QG_QAG = "QG_QAG"

class ExperimentDG(Enum):
    DG_ONLY = "DG"
    DAG_ONLY = "DAG"
    DG_DAG = "DG_DAG"

class Metadata(Enum):
    QG = "QG"
    QAG = "QAG"
    DG = "DG"
    DAG = "DAG"