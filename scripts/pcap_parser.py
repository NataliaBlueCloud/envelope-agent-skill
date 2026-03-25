"""
PCAP Parser for Envelope Agent

Read packet capture files and extract delays for envelope fitting.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


def read_pcap_scapy(pcap_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read PCAP file using scapy.
    
    Args:
        pcap_file: Path to .pcap or .pcapng file
        
    Returns:
        (timestamps, packet_sizes) as numpy arrays
        
    Raises:
        ImportError: If scapy is not installed
        FileNotFoundError: If pcap file doesn't exist
    """
    try:
        from scapy.all import rdpcap
    except ImportError:
        raise ImportError(
            "scapy is required for PCAP reading. "
            "Install with: pip install scapy"
        )
    
    pcap_path = Path(pcap_file)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_file}")
    
    packets = rdpcap(str(pcap_path))
    
    timestamps = np.array([float(pkt.time) for pkt in packets])
    packet_sizes = np.array([len(pkt) for pkt in packets])
    
    return timestamps, packet_sizes


def read_pcap_dpkt(pcap_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read PCAP file using dpkt (faster, no scapy dependency).
    
    Args:
        pcap_file: Path to .pcap file
        
    Returns:
        (timestamps, packet_sizes) as numpy arrays
        
    Raises:
        ImportError: If dpkt is not installed
    """
    try:
        import dpkt
    except ImportError:
        raise ImportError(
            "dpkt is required for PCAP reading. "
            "Install with: pip install dpkt"
        )
    
    pcap_path = Path(pcap_file)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_file}")
    
    timestamps = []
    packet_sizes = []
    
    with open(pcap_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, buf in pcap:
            timestamps.append(ts)
            packet_sizes.append(len(buf))
    
    return np.array(timestamps), np.array(packet_sizes)


def read_pcap_auto(pcap_file: str, prefer: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
    """
    Read PCAP file using best available library.
    
    Args:
        pcap_file: Path to .pcap file
        prefer: 'scapy', 'dpkt', or 'auto' (tries both)
        
    Returns:
        (timestamps, packet_sizes) as numpy arrays
    """
    if prefer == 'scapy':
        return read_pcap_scapy(pcap_file)
    elif prefer == 'dpkt':
        return read_pcap_dpkt(pcap_file)
    
    # Auto: try dpkt first (faster), fallback to scapy
    try:
        return read_pcap_dpkt(pcap_file)
    except ImportError:
        return read_pcap_scapy(pcap_file)


def filter_by_size(timestamps: np.ndarray,
                   packet_sizes: np.ndarray,
                   min_size: Optional[int] = None,
                   max_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter packets by size (e.g., remove ACKs, keep only data).
    
    Args:
        timestamps: Packet timestamps
        packet_sizes: Packet sizes
        min_size: Minimum packet size to keep
        max_size: Maximum packet size to keep
        
    Returns:
        Filtered (timestamps, packet_sizes)
    """
    mask = np.ones(len(packet_sizes), dtype=bool)
    
    if min_size is not None:
        mask &= packet_sizes >= min_size
    if max_size is not None:
        mask &= packet_sizes <= max_size
    
    return timestamps[mask], packet_sizes[mask]


def compute_inter_arrival_stats(timestamps: np.ndarray) -> Dict:
    """
    Compute inter-arrival time statistics from timestamps.
    
    Args:
        timestamps: Packet timestamps in seconds
        
    Returns:
        Dictionary with inter-arrival stats
    """
    inter_arrivals = np.diff(timestamps)
    
    return {
        'mean_ia_s': np.mean(inter_arrivals),
        'std_ia_s': np.std(inter_arrivals),
        'min_ia_s': np.min(inter_arrivals),
        'max_ia_s': np.max(inter_arrivals),
        'median_ia_s': np.median(inter_arrivals),
        'arrival_rate_pps': 1.0 / np.mean(inter_arrivals) if np.mean(inter_arrivals) > 0 else 0
    }


def summarize_pcap(timestamps: np.ndarray, packet_sizes: np.ndarray) -> Dict:
    """
    Generate summary statistics for PCAP data.
    
    Args:
        timestamps: Packet timestamps
        packet_sizes: Packet sizes
        
    Returns:
        Summary dictionary
    """
    duration = timestamps[-1] - timestamps[0]
    total_bytes = np.sum(packet_sizes)
    
    # Compute SCV (squared coefficient of variation)
    mean_size = np.mean(packet_sizes)
    std_size = np.std(packet_sizes)
    scv = (std_size / mean_size) ** 2 if mean_size > 0 else 0
    
    ia_stats = compute_inter_arrival_stats(timestamps)
    
    return {
        'num_packets': len(timestamps),
        'duration_s': duration,
        'total_bytes': int(total_bytes),
        'avg_packet_bytes': mean_size,
        'std_packet_bytes': std_size,
        'scv': scv,
        'avg_throughput_mbps': (total_bytes * 8 / duration / 1e6) if duration > 0 else 0,
        'inter_arrival': ia_stats
    }


# Convenience exports
__all__ = [
    'read_pcap_scapy',
    'read_pcap_dpkt', 
    'read_pcap_auto',
    'filter_by_size',
    'compute_inter_arrival_stats',
    'summarize_pcap'
]
