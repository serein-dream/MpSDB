
def top_model_fp_bp(model, optimizer, middle_output):
    model.train()
    pred = model(middle_output).cpu()
    optimizer.zero_grad()


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def dcn_fp(epoch, features, prob, network):
    # treatment_pred[0] -> y1
    # treatment_pred[1] -> y0

    if epoch % 2 == 0:
        # train treated

        network.hidden1_Y1.weight.requires_grad = True
        network.hidden1_Y1.bias.requires_grad = True
        network.hidden2_Y1.weight.requires_grad = True
        network.hidden2_Y1.bias.requires_grad = True
        network.out_Y1.weight.requires_grad = True
        network.out_Y1.bias.requires_grad = True

        network.hidden1_Y0.weight.requires_grad = False
        network.hidden1_Y0.bias.requires_grad = False
        network.hidden2_Y0.weight.requires_grad = False
        network.hidden2_Y0.bias.requires_grad = False
        network.out_Y0.weight.requires_grad = False
        network.out_Y0.bias.requires_grad = False

        treatment_pred = network(features, prob)
        predicted_ite = treatment_pred[0] - treatment_pred[1]

        return predicted_ite

    elif epoch % 2 == 1:
        # train controlled

        network.hidden1_Y1.weight.requires_grad = False
        network.hidden1_Y1.bias.requires_grad = False
        network.hidden2_Y1.weight.requires_grad = False
        network.hidden2_Y1.bias.requires_grad = False
        network.out_Y1.weight.requires_grad = False
        network.out_Y1.bias.requires_grad = False

        network.hidden1_Y0.weight.requires_grad = True
        network.hidden1_Y0.bias.requires_grad = True
        network.hidden2_Y0.weight.requires_grad = True
        network.hidden2_Y0.bias.requires_grad = True
        network.out_Y0.weight.requires_grad = True
        network.out_Y0.bias.requires_grad = True

        treatment_pred = network(features, prob)
        predicted_ite = treatment_pred[0] - treatment_pred[1]

        return predicted_ite



