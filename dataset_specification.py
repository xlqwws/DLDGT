from typing import List, Optional


class DatasetSpecification:
    def __init__(self, include_fields: List[str], categorical_fields: List[str], class_column: str, benign_label: str,
                 test_column: Optional[str] = None):
        """
        Define specific dataset formats
        :param include_fields: Fields used as classification criteria
        :param categorical_fields: Fields to be treated as categorical
        :param class_column: Column name containing traffic categories
        :param benign_label: Label e.g., Benign or 0
        :param test_column: Indicates whether the row belongs to the training or test set
        """
        self.include_fields: List[str] = include_fields
        self.categorical_fields: List[str] = categorical_fields
        self.class_column = class_column
        self.benign_label = benign_label
        self.test_column: Optional[str] = test_column


class NamedDatasetSpecifications:
    """
    Example specifications of some common datasets
    """

    CICIDS2017 = DatasetSpecification(

        include_fields=[' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
                        ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Mean',
                        ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Mean',
                        ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std',
                        ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
                        ' Fwd IAT Max', ' Fwd IAT Min',
                        'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Fwd Header Length',
                        ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', ' Max Packet Length',
                        ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', ' Average Packet Size',
                        ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length', 'Subflow Fwd Packets',
                        ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                        ' Init_Win_bytes_backward',
                        ' act_data_pkt_fwd', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean',
                        ' Idle Std', ' Idle Max', ' Idle Min'],
        categorical_fields=[' Destination Port', ' Flow Duration', ' Total Length of Bwd Packets', 'Flow Bytes/s',
                            ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Max', 'Fwd IAT Total', 'Bwd IAT Total',
                            ' Packet Length Variance', 'Fwd Packets/s', ' Bwd Packets/s', ' Init_Win_bytes_backward'],
        class_column=" Label",
        benign_label="BENIGN"
    )

    CICIDS2018 = DatasetSpecification(

        include_fields=['Dst Port', 'Protocol', 'Timestamp', 'Flow Duration',
                        'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts',
                        'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
                        'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
                        'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std',
                        'Flow IAT Max', 'Flow IAT Min',
                        'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
                        'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
                        'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
                        'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
                        'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
                        'Pkt Len Std',
                        'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt',
                        'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt','Down/Up Ratio','Pkt Size Avg','Fwd Seg Size Avg','Bwd Seg Size Avg','Fwd Byts/b Avg','Fwd Pkts/b Avg',
                        'Fwd Blk Rate Avg','Bwd Byts/b Avg','Bwd Pkts/b Avg','Bwd Blk Rate Avg','Subflow Fwd Pkts',
                        'Subflow Fwd Byts','Subflow Bwd Pkts','Subflow Bwd Byts',
                        'Init Fwd Win Byts','Init Bwd Win Byts','Fwd Act Data Pkts','Fwd Seg Size Min','Active Mean','Active Std',
                        'Active Max','Active Min','Idle Mean','Idle Std','Idle Max','Idle Min'

                        ],
        categorical_fields=['Dst Port', 'Flow Duration', 'TotLen Bwd Pkts', 'Flow IAT Min',
                            'Flow IAT Max', 'Fwd IAT Mean', 'Pkt Len Max', 'Fwd Header Len', 'FIN Flag Cnt',
                            'Bwd Blk Rate Avg', 'Idle Std', 'Pkt Len Var', 'Bwd URG Flags'],
        class_column=" Label",
        benign_label="BENIGN"
    )

    nsl_kdd = DatasetSpecification(
        include_fields=['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
                        'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                        'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
                        'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
                        'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'],
        categorical_fields=['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login'],
        benign_label='normal',
        class_column='label',
        test_column='is_test'
    )

    UnknownAttack = DatasetSpecification(

        include_fields=[' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
                        ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Mean',
                        ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Mean',
                        ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std',
                        ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
                        ' Fwd IAT Max', ' Fwd IAT Min',
                        'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Fwd Header Length',
                        ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', ' Max Packet Length',
                        ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', ' Average Packet Size',
                        ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length', 'Subflow Fwd Packets',
                        ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                        ' Init_Win_bytes_backward',
                        ' act_data_pkt_fwd', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean',
                        ' Idle Std', ' Idle Max', ' Idle Min'],
        categorical_fields=[' Destination Port', ' Flow Duration', ' Total Length of Bwd Packets', 'Flow Bytes/s',
                            ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Max', 'Fwd IAT Total', 'Bwd IAT Total',
                            ' Packet Length Variance', 'Fwd Packets/s', ' Bwd Packets/s', ' Init_Win_bytes_backward'],
        class_column=" Label",
        benign_label="BENIGN"
    )

