def train(model, device, train_loader,lr_schedule, optimizer,train_losses,train_acc,tqdm,criterion,smooth_f,diverge_th):
  model.train()
  history = {"lr": [], "loss": []}
  iter=0
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  best_loss=None
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    # pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    # train_acc.append(100*correct/processed)
    lr_schedule.step()
    lrn=lr_schedule.get_lr()[0]
    history["lr"].append(lrn)
    # Track the best loss and smooth it if smooth_f is specified
    if iter == 0:
                best_loss = loss
                best_lern=  lrn
    else:
                if smooth_f > 0:
                    loss = smooth_f * loss + (1 - smooth_f) * history["loss"][-1]
                if loss < best_loss:
                    best_loss = loss
                    best_lern=  lrn

    # Check if the loss has diverged; if it has, stop the test
    history["loss"].append(loss)
    if loss > diverge_th * best_loss:
                print("Stopping early, the loss has diverged")
                break
    iter=iter+1
  return best_lern,best_loss,history
      