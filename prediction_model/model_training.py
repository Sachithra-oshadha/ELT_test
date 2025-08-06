from imports import *

def train_model(model, train_loader: DataLoader, val_loader: DataLoader, logger: logging.Logger, num_epochs: int = 10, patience: int = 3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            break
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        # Validation loop
        model.eval()
        val_loss = 0
        predictions, actuals = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs.squeeze(), batch_y.squeeze()).item() * batch_x.size(0)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())

        # Post-epoch processing
        val_loss /= len(val_loader.dataset)
        train_loss /= len(train_loader.dataset)
        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)
        r2 = r2_score(actuals.flatten(), predictions.flatten())
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R² Score: {r2:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True
                logger.info(f"No improvement in validation loss for {patience} epochs")

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, best_val_loss, r2