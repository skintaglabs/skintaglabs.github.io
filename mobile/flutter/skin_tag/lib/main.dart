import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'theme/app_theme.dart';
import 'services/model_service.dart';
import 'screens/disclaimer_screen.dart';
import 'screens/camera_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const SkinTagApp());
}

class SkinTagApp extends StatelessWidget {
  const SkinTagApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => AppState(),
      child: MaterialApp(
        title: 'SkinTag',
        theme: AppTheme.lightTheme,
        darkTheme: AppTheme.darkTheme,
        themeMode: ThemeMode.system,
        home: const AppNavigator(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}

/// Manages app navigation based on disclaimer acceptance
class AppNavigator extends StatelessWidget {
  const AppNavigator({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<AppState>(
      builder: (context, appState, child) {
        if (!appState.hasAcceptedDisclaimer) {
          return const DisclaimerScreen();
        }

        if (appState.modelLoadError != null) {
          return ModelErrorScreen(error: appState.modelLoadError!);
        }

        if (!appState.modelLoaded) {
          return const LoadingScreen();
        }

        return const CameraScreen();
      },
    );
  }
}

/// Loading screen while model loads
class LoadingScreen extends StatelessWidget {
  const LoadingScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 20),
            Text(
              'Loading AI model...',
              style: TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}

/// Error screen if model fails to load
class ModelErrorScreen extends StatelessWidget {
  final String error;

  const ModelErrorScreen({super.key, required this.error});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.warning_amber_rounded,
                size: 64,
                color: Colors.orange,
              ),
              const SizedBox(height: 20),
              const Text(
                'Model Load Error',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 12),
              Text(
                error,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: () {
                  context.read<AppState>().loadModel();
                },
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Global application state
class AppState extends ChangeNotifier {
  bool hasAcceptedDisclaimer = false;
  bool modelLoaded = false;
  String? modelLoadError;

  ModelService? _modelService;

  AppState() {
    loadModel();
  }

  ModelService? get modelService => _modelService;

  void acceptDisclaimer() {
    hasAcceptedDisclaimer = true;
    notifyListeners();
  }

  Future<void> loadModel() async {
    try {
      modelLoadError = null;
      _modelService = ModelService();
      await _modelService!.loadModel();
      modelLoaded = true;
    } catch (e) {
      modelLoadError = e.toString();
      modelLoaded = false;
    }
    notifyListeners();
  }
}
