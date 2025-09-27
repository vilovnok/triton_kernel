import torch

import wandb
import time

from .utils import *


def train(model, train_data, val_data, batch_size, seq_length, learning_rate, num_epochs, wandb_project=None, wandb_run_name=None, timeout=None):
    
    if wandb_project:
        wandb.init(project=wandb_project, name=wandb_run_name, settings=wandb.Settings(init_timeout=timeout))
        wandb.config.update({
            "batch_size": batch_size,
            "seq_length": seq_length,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "model_dim": model.dim,
            "num_heads": model.num_heads,
            "num_layers": model.num_layers
        })

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    iter_num = 0
    best_val_loss = float('inf')
    val_losses = []

    model.train()
    t0 = time.time()
    for epoch in range(num_epochs):
        for _ in range(100):
            iter_num += 1

            t_start = time.time()

            xb, yb = get_batch('train', model, train_data, val_data, seq_length, batch_size)
            t_data = time.time()

            logits = model(xb)
            t_forward = time.time()

            loss = model.compute_loss(logits, yb)
            t_loss = time.time()

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"Warning: NaN or Inf detected in loss at iteration {iter_num}")
                print(f"Logits min: {logits.min()}, max: {logits.max()}")
                print(f"Target min: {yb.min()}, max: {yb.max()}")
                continue

            optimizer.zero_grad()
            loss.backward()
            t_backward = time.time()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            t_optim = time.time()

            if iter_num % 10 == 0:
                dt = t_optim - t_start
                dt_data = t_data - t_start
                dt_forward = t_forward - t_data
                dt_loss = t_loss - t_forward
                dt_backward = t_backward - t_loss
                dt_optim = t_optim - t_backward
                mfu = estimate_mfu(model, dt, batch_size)
                
                print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, mfu {mfu*100:.2f}%")

                if wandb_project:
                    wandb.log({
                        "train/loss": loss.item(),
                        "iter": iter_num,
                        "mfu": mfu*100
                    })
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _ in range(50): 
                xb, yb = get_batch('val', model, train_data, val_data, seq_length, batch_size)
                logits = model(xb)
                val_loss += model.compute_loss(logits, yb).item()
        val_loss /= 50
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")


        if wandb_project:
            wandb.log({
                "val/loss": val_loss,
                "epoch": epoch+1,
                "learning_rate": scheduler.get_last_lr()[0]
            })

        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'Checkpoints/nanoGPT_cpkt.pth')
            print(f"Saved checkpoint for validation loss: {best_val_loss:.4f}")

        model.train()

    return model, val_losses