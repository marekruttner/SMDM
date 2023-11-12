import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    // Ensure the widgets binding is initialized
    WidgetsFlutterBinding.ensureInitialized();

    // Create and run the app state
    final appState = AppState();
    appState.startContinuousFunction();

    return MaterialApp(
      title: 'Click Tracker',
      home: MyHomePage(appState: appState),
    );
  }
}

class AppState {
  List<Coordinate> coordinates = [];
  late StreamController<List<Coordinate>> _controller;
  late File _file;

  AppState() {
    _controller = StreamController<List<Coordinate>>.broadcast();
    _file = File('click_coordinates.csv');
  }

  void startContinuousFunction() {
    WidgetsBinding.instance?.addPostFrameCallback((_) {
      _continuousFunction();
    });
  }

  void _continuousFunction() {
    _controller.add(coordinates);

    // Schedule the function to run again after the next frame
    WidgetsBinding.instance?.addPostFrameCallback((_) {
      _continuousFunction();
    });
  }

  void recordClick(Coordinate coordinate) {
    coordinates.add(coordinate);
    _writeToFile();
  }

  void _writeToFile() {
    final csvContent = coordinates.map((coord) => '${coord.x},${coord.y}\n').join();
    _file.writeAsStringSync(csvContent);
  }

  Stream<List<Coordinate>> get coordinateStream => _controller.stream;
}

class MyHomePage extends StatelessWidget {
  final AppState appState;

  const MyHomePage({Key? key, required this.appState}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTapDown: (details) {
        // Get the coordinates of the tap
        final RenderBox renderBox = context.findRenderObject() as RenderBox;
        final coordinates = renderBox.globalToLocal(details.globalPosition);
        appState.recordClick(Coordinate(x: coordinates.dx, y: coordinates.dy));
      },
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Click Tracker'),
        ),
        body: Center(
          child: StreamBuilder<List<Coordinate>>(
            stream: appState.coordinateStream,
            builder: (context, snapshot) {
              if (snapshot.hasData) {
                final coordinates = snapshot.data!;
                return Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text('Coordinates: $coordinates'),
                  ],
                );
              } else {
                return const CircularProgressIndicator();
              }
            },
          ),
        ),
      ),
    );
  }
}

class Coordinate {
  final double x;
  final double y;

  Coordinate({required this.x, required this.y});

  @override
  String toString() {
    return '($x, $y)';
  }
}
