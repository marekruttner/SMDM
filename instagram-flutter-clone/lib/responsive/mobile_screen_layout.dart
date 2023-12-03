import 'dart:html';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:instagram_clone_flutter/utils/colors.dart';
import 'package:instagram_clone_flutter/utils/global_variable.dart';

import '../models/post.dart';

import 'dart:io';
import 'package:csv/csv.dart';
import 'package:file/file.dart';

import 'dart:html' as html;

class MobileScreenLayout extends StatefulWidget {
  const MobileScreenLayout({Key? key}) : super(key: key);

  @override
  State<MobileScreenLayout> createState() => _MobileScreenLayoutState();
}

class _MobileScreenLayoutState extends State<MobileScreenLayout> {
  int _page = 0;
  late PageController pageController;

  List<Post> get posts => []; // for tabs animation

  @override
  void initState() {
    super.initState();
    pageController = PageController();
  }

  @override
  void dispose() {
    super.dispose();
    pageController.dispose();
  }

  void onPageChanged(int page) {
    setState(() {
      _page = page;
    });
  }

  void navigationTapped(int page) {
    // Animating Page
    pageController.jumpToPage(page);
  }

  void onTapDown(TapDownDetails details) {
    // Get the coordinates of the tap
    final RenderBox renderBox = context.findRenderObject() as RenderBox;
    final coordinates = renderBox.globalToLocal(details.globalPosition);
    final postHeight = MediaQuery.of(context).size.height * 0.7;
    final index = (coordinates.dy / postHeight).floor();

    // Print tap coordinates
    print('Tap Coordinates: ${coordinates.dx}, ${coordinates.dy} index of post: $index');

    // Get the UID of the post associated with the tapped area
    //String postUid = getPostUidForCoordinates(coordinates, posts);

    // Log the post UID
    //print('Post UID: $postUid');

  }




  Object getPostUidForCoordinates(Offset coordinates, List<Post> posts) {
    // Assuming each post has a fixed height, calculate the index of the tapped post
    final postHeight = MediaQuery.of(context).size.height * 0.7;
    final index = (coordinates.dy / postHeight).floor();

    // Print for debugging
    print('Tap Coordinates: ${coordinates.dy}');
    print('Post Height: $postHeight');
    print('Calculated Index: $index');

    // Check if the index is valid
    if (index >= 0 && index < posts.length) {
      // Access the 'postId' field directly from the post
      return posts[index].postId;
    } else {
      return 'NO POST'; // No post found for the given coordinates
    }
    return index;
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: GestureDetector(
        onTapDown: onTapDown,
        child: PageView(
          controller: pageController,
          onPageChanged: onPageChanged,
          children: homeScreenItems,
        ),
      ),
      bottomNavigationBar: CupertinoTabBar(
        backgroundColor: mobileBackgroundColor,
        items: <BottomNavigationBarItem>[
          BottomNavigationBarItem(
            icon: Icon(
              Icons.home,
              color: (_page == 0) ? primaryColor : secondaryColor,
            ),
            label: '',
            backgroundColor: primaryColor,
          ),
          BottomNavigationBarItem(
              icon: Icon(
                Icons.search,
                color: (_page == 1) ? primaryColor : secondaryColor,
              ),
              label: '',
              backgroundColor: primaryColor),
          BottomNavigationBarItem(
              icon: Icon(
                Icons.add_circle,
                color: (_page == 2) ? primaryColor : secondaryColor,
              ),
              label: '',
              backgroundColor: primaryColor),
          BottomNavigationBarItem(
            icon: Icon(
              Icons.favorite,
              color: (_page == 3) ? primaryColor : secondaryColor,
            ),
            label: '',
            backgroundColor: primaryColor,
          ),
          BottomNavigationBarItem(
            icon: Icon(
              Icons.person,
              color: (_page == 4) ? primaryColor : secondaryColor,
            ),
            label: '',
            backgroundColor: primaryColor,
          ),
        ],
        onTap: navigationTapped,
        currentIndex: _page,
      ),
    );
  }
}
