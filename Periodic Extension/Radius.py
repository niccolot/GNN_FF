import torch
import numpy as np
import scipy.spatial
from PeriodicTree import PeriodicCKDTree



def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        r (float): The radius.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import radius

    .. testcode::


        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch_x = torch.tensor([0, 0, 0, 0])
        >>> y = torch.Tensor([[-1, 0], [1, 0]])
        >>> batch_y = torch.tensor([0, 0])
        >>> assign_index = radius(x, y, 1.5, batch_x, batch_y)
    """

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)


    x = torch.cat([x, 2 * r * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * r * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

    tree = scipy.spatial.cKDTree(x.detach().numpy())
    _, col = tree.query(
        y.detach().numpy(), k=max_num_neighbors, distance_upper_bound=r + 1e-8)
    col = [torch.from_numpy(c).to(torch.long) for c in col]
    row = [torch.full_like(c, i) for i, c in enumerate(col)]
    row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)
    mask = col < int(tree.n)
    return torch.stack([row[mask], col[mask]], dim=0)



def radius_graph(x,
                 r,
                 batch=None,
                 loop=False,
                 max_num_neighbors=32,
                 flow='source_to_target'):
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import radius_graph

    .. testcode::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch = torch.tensor([0, 0, 0, 0])
        >>> edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """

    assert flow in ['source_to_target', 'target_to_source']
    row, col = radius(x, x, r, batch, batch, max_num_neighbors + 1)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)


def radius_periodic(x, y, r,bounds, batch_x=None, batch_y=None, max_num_neighbors=32):
    r"""Finds for each element in :obj:`y` all points in :obj:`x` within
    distance :obj:`r`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{Y} \in \mathbb{R}^{M \times F}`.
        r (float): The radius.
        bounds (array): array where each element is the lenght of the periodic
            box (its dimention must match the box one)
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import radius

    .. testcode::


        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch_x = torch.tensor([0, 0, 0, 0])
        >>> y = torch.Tensor([[-1, 0], [1, 0]])
        >>> batch_y = torch.tensor([0, 0])
        >>> assign_index = radius_periodic(x, y, 1.5, bounds, batch_x, batch_y)
    """



    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    x = torch.cat([x, 2 * r * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * r * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

    tree = PeriodicCKDTree(bounds,x.detach().numpy()) #lui è il comando per il tree periodico
    _, col = tree.query(
        y.detach().numpy(), k=max_num_neighbors, distance_upper_bound=r + 1e-8)
    col = [torch.from_numpy(c).to(torch.long) for c in col]
    row = [torch.full_like(c, i) for i, c in enumerate(col)]
    row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)
    mask = col < int(tree.n)
    return torch.stack([row[mask], col[mask]], dim=0)




#---------------------------------------------------------
def radius_graph_pbc(x,
                 r,
                 bounds,
                 batch=None,
                 loop=False,
                 max_num_neighbors=32,
                 flow='source_to_target'):
    r"""Computes graph edges to all points within a given distance.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        r (float): The radius.
        bounds (array): array where each element is the lenght of the periodic
            box (its dimention must match the box one)
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            return for each element in :obj:`y`. (default: :obj:`32`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import radius_graph

    .. testcode::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch = torch.tensor([0, 0, 0, 0])
        >>> edge_index = radius_graph(x, r=1.5, batch=batch, loop=False)
    """
    # Added since the bounds must be an array with dimention 4 since positions have one extra coordinate
    
    bounds = np.array(bounds)
    bounds = np.append(bounds, bounds[0][0])
    print(np.shape(bounds))    
    assert flow in ['source_to_target', 'target_to_source']
    row, col = radius_periodic(x, x, r, bounds, batch, batch, max_num_neighbors + 1)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)

def distance_pbc(pos1, pos2, dimensions):
    """
    This function takes as input two tensor :obj:'pos1' and :obj:'pos2' and gives a tensor that represent the periodic distance between
    the two input tesors

    Parameters
    ----------
        - pos1 (Tensor): tensor typically (1,3) which represent
            the position of the first node
        - pos2 (Tensor): tensor typically (1,3) which represent
            the position of the second node
        - dimensions (tuple): a tuple containg the length of the box in each dimension
    
    Results
    ---------
        - distance (Tensor)
    """
    pos1 = pos1.numpy()
    pos2 = pos2.numpy()
    delta = np.abs(pos1 - pos2)
    delta = np.where(delta > 0.5 * np.array(dimensions), delta - dimensions, delta)
    return np.sqrt((delta ** 2).sum(axis=-1))

