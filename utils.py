from functools import reduce

import torch
from loguru import logger
from tqdm import tqdm

from _utils import mean_average_precision


def predict(net, dataloader, out_idx=None, use_sign=True, verbose=True):
    device = next(net.parameters()).device
    net.eval()

    data_iter = tqdm(dataloader, desc="Extracting features") if verbose else dataloader
    embs, labs = [], []

    # for imgs, labs, _ in tqdm(dataloader):
    # for imgs, txts, labs, _ in tqdm(dataloader): # CrossModal
    for batch in data_iter:
        with torch.no_grad():
            out = net(batch[0].to(device), batch[-2].to(device))
        if out_idx is None:
            embs.append(out)
        elif isinstance(out_idx, list):
            # CMCL
            rst = reduce(lambda d, key: d[key], out_idx, out)
            embs.append(rst)
        else:
            embs.append(out[out_idx])
        labs.append(batch[-2])
    return torch.cat(embs).sign() if use_sign else torch.cat(embs), torch.cat(labs).to(device)


def validate(args, query_loader, dbase_loader, early_stopping, epoch, **kwargs):
    out_idx = kwargs.pop("out_idx", None)
    verbose = kwargs.pop("verbose", True)
    map_fnc = kwargs.pop("map_fnc", mean_average_precision)

    qB, qL = predict(kwargs["model"], query_loader, out_idx=out_idx, verbose=verbose)
    rB, rL = predict(kwargs["model"], dbase_loader, out_idx=out_idx, verbose=verbose)
    map_v = map_fnc(qB, rB, qL, rL, args.topk)
    map_k = "" if args.topk is None else f"@{args.topk}"

    del qB, rB, qL, rL
    torch.cuda.empty_cache()

    map_o = early_stopping.best_map
    early_stopping(epoch, map_v.item(), **kwargs)
    logger.info(
        f"[Evaluating][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][best-mAP{map_k}:{map_o:.4f}][mAP{map_k}:{map_v:.4f}][count:{early_stopping.counter}]"
    )
    return early_stopping.early_stop
