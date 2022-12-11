def parseargs():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CiteSeer')
    parser.add_argument('--experiment', type=str, default='fixed') #'fixed', 'random', 'few'
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--cratio', '--coarsening_ratio',type=float, default=0.5)
    parser.add_argument('--coarsening_method', type=str, default='NONE')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--levels', type=int,default=1)
    # Set our arguments as 'args'.
    args = parser.parse_args()
    
    return( args.hidden,
            args.layers,
            args.optimizer,
            args.lr,
            args.epochs,
            args.cratio,
            args.levels,
            args.coarsening_method,
            args.dataset)

def validate(model, data, crit):
    model.eval()
    yhat = model(data).argmax(dim=1)
    correct = (yhat == data.y)
    accuracy = correct.sum() / len(correct)
    return accuracy


def train_epoch(model, data, optimizer, loss_fn):
    model.train()
    optimizer.zero_grad()
    yhat = model(data)
    loss = loss_fn(yhat, data.y)
    loss.backward()
    optimizer.step()

def DMDstep(model, weights, r, pred_step, params=None):
    if params is None:
        params = [i for i in range(len(weights))]
        
    for i,(W,param) in tqdm(enumerate(zip(weights,model.parameters())), total=len(params)):
        if i in params:
            start = time.time()
            M = W.reshape(W.shape[0], np.prod(W.shape[1:])).T
            new_weight = DMD4cast(M, r, pred_step)[:,-1].T
            new_weight = new_weight.reshape(W[0].shape)

            param.data = Tensor(new_weight)
            print(f"Step {i}: {time.time()-start}")

def train_level(model, TGG_D, optimizer, loss_fn, level, m=None, pred_step=None, r=2, epochs=20):
    accuracy = []
    model.train()
    if m is None:
        m = epochs+1
    data = TGG_D[level].to(device)
    data.x = F.normalize(data.x,p=1)
    
    weights = [ np.empty(np.append(1, param.shape)) for param in model.parameters()]
    for i,param in enumerate(model.parameters()):
        weights[i][0] = param.detach()


    for epoch in tqdm(range(1,epochs+1)):
        if epoch % m == 0:
            print("DMD")
            DMDstep(model, weights, r, pred_step, params=None)
            weights = [ np.empty(np.append(1, param.shape)) for param in model.parameters()]
        train_epoch(model, data, optimizer, loss_fn)
        for i,param  in enumerate(model.parameters()):
            weights[i] = np.append(weights[i], param.detach().reshape(np.append(1,weights[i].shape[1:]).tolist()),axis=0)
        