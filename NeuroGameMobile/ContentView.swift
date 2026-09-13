import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = SimulationViewModel()
    @State private var showsBrain = true

    var body: some View {
        VStack(spacing: 0) {
            SimulationCanvasView(viewModel: viewModel)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            ControlBarView(viewModel: viewModel, showsBrain: $showsBrain)

            if showsBrain {
                NeuralActivityView(snapshot: viewModel.brainSnapshot)
                    .frame(height: 230)
                    .padding(.horizontal, 12)
                    .padding(.bottom, 10)
            }
        }
        .background(Color(red: 0.06, green: 0.07, blue: 0.075))
        .foregroundStyle(.white)
        .onAppear { viewModel.start() }
        .onDisappear { viewModel.stop() }
    }
}

private struct ControlBarView: View {
    @ObservedObject var viewModel: SimulationViewModel
    @Binding var showsBrain: Bool

    var body: some View {
        HStack(spacing: 12) {
            Button {
                viewModel.toggleRunning()
            } label: {
                Image(systemName: viewModel.isRunning ? "pause.fill" : "play.fill")
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(.borderedProminent)

            Button {
                viewModel.reset()
            } label: {
                Image(systemName: "arrow.clockwise")
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(.bordered)

            Button {
                viewModel.changeMap()
            } label: {
                Image(systemName: "map")
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(.bordered)

            Button {
                showsBrain.toggle()
            } label: {
                Image(systemName: showsBrain ? "brain.head.profile.fill" : "brain.head.profile")
                    .frame(width: 32, height: 32)
            }
            .buttonStyle(.bordered)

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("먹이 \(viewModel.foodEaten)")
                    Spacer(minLength: 8)
                    Text("맵 \(viewModel.mapNumber)")
                }
                .font(.caption.weight(.semibold))
                ProgressView(value: viewModel.energy, total: 100).tint(.mint)
            }

        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(Color(red: 0.095, green: 0.105, blue: 0.11))
    }
}
