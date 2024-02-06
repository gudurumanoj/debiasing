seed = 42
torch.manual_seed(seed)

time_per_run = []
acc_per_run = []

for i in range(num_runs):
    # Define the Model
    model = LeNet()
    model = model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    loss_fn_meta = nn.CrossEntropyLoss(reduction='none')

    # Train the model
    model.train()
    start_time = time.time()
    for epoch in tqdm(range(epochs)):
        # Train loop
        for images, labels in train_dataloader:
            
            images = images.to(device)
            labels = labels.to(device)

            # meta_net = get_cifar10_model()
            meta_net = LeNet()
            meta_net.load_state_dict(model.state_dict())

            meta_net = meta_net.to(device)

            optimizer_meta = torch.optim.Adam(meta_net.parameters())

            meta_net.train()
            
            y_f_hat = meta_net(images)
            cost = loss_fn_meta(y_f_hat, labels)
            eps = torch.zeros(cost.size(), requires_grad=True).to(device)
            l_f_meta = torch.sum(cost*eps)

            # meta_net.zero_grad()
            optimizer_meta.zero_grad()
            eps.retain_grad()
            l_f_meta.backward()
            optimizer_meta.step()

            meta_net.eval()

            # grads = torch.autograd.grad(l_f_meta, (meta_net.parameters()), create_graph=True)
            # meta_net.update_params(lr, source_params=grads)
            
            val_images, val_labels = next(iter(subset_dataloader))
            # val_images, val_labels = next(iter(val_dataloader))
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)

            y_g_hat = meta_net(val_images)
            l_g_meta = loss_fn(y_g_hat, val_labels)

            # grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True)[0]
            # grad_eps = torch.autograd.grad(l_g_meta, eps, only_inputs=True, allow_unused=True)[0]
            # print(grad_eps)

            with torch.no_grad():
                optimizer_meta.zero_grad()
                l_g_meta.backward()
                grad_eps = eps.grad
            
            # print(grad_eps)
            w_tilde = torch.clamp(grad_eps,min=0)
            # w_tilde = torch.clamp(-grad_eps,min=0)
            norm_c = torch.sum(w_tilde)

            if norm_c != 0:
                w = w_tilde / norm_c
            else:
                w = w_tilde
            
            # print(w)
            # break
            # Forward Pass
            outputs = model(images)
            loss = loss_fn_meta(outputs, labels)
            loss = torch.sum(loss*w)
            
            # Backward pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    time_taken = time.time() - start_time   
    time_per_run.append(time_taken)  
    print("--- %s seconds ---" % (time_taken))

    # Evaluate on test set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    acc_per_run.append(accuracy)
    print(f"Accuracy: {accuracy:.4f}")